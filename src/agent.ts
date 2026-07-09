import WebSocket, { WebSocketServer } from "ws";
import type { Server } from "http";
import fsPromises from "fs/promises";
import fs from "fs";
import { execSync } from "child_process";

const ZEN_API_KEY = process.env.ZEN_API_KEY;
const ZEN_MODEL = process.env.ZEN_MODEL || "big-pickle";
const ZEN_API_URL = "https://opencode.ai/zen/v1/chat/completions";
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_VOICE_ID || "pNInz6obpgDQGcFmaJgB";
const ELEVENLABS_API_URL = `https://api.elevenlabs.io/v1/text-to-speech/${ELEVENLABS_VOICE_ID}`;

const SILENCE_TIMEOUT_MS = 800;
const SAMPLE_RATE = 8000;
const BYTES_PER_SAMPLE = 2;

const BIAS = 0x84;
const CLIP = 32635;

const pcmToUlawLut = new Uint8Array(65536);
const ulawToPcmLut = new Int16Array(256);

(function initTables() {
  for (let i = 0; i < 256; i++) {
    const ulaw = ~i;
    const sign = ulaw & 0x80;
    const exponent = (ulaw >> 4) & 0x07;
    const mantissa = ulaw & 0x0f;
    const sample = ((mantissa << 3) + BIAS) << (exponent + 3);
    ulawToPcmLut[i] = sign ? (CLIP - sample) : (sample - CLIP);
  }

  for (let s = -32768; s <= 32767; s++) {
    const sign = s < 0 ? 0x80 : 0x00;
    let sample = Math.abs(s);
    if (sample > CLIP) sample = CLIP;
    let exponent = 7;
    let mask = 0x4000;
    while (!(sample & mask) && exponent > 0) {
      mask >>= 1;
      exponent--;
    }
    const mantissa = (sample >> (exponent + 3)) & 0x0f;
    pcmToUlawLut[s & 0xffff] = ~(sign | (exponent << 4) | mantissa);
  }
})();

function ulawToPcm(buf: Buffer): Buffer {
  const out = Buffer.alloc(buf.length * 2);
  for (let i = 0; i < buf.length; i++) {
    out.writeInt16LE(ulawToPcmLut[buf[i]], i * 2);
  }
  return out;
}

function pcmToUlaw(buf: Buffer): Buffer {
  const out = Buffer.alloc(Math.floor(buf.length / 2));
  for (let i = 0; i < out.length; i++) {
    out[i] = pcmToUlawLut[(buf.readInt16LE(i * 2) + 32768) & 0xffff];
  }
  return out;
}

function createWavHeader(dataLen: number): Buffer {
  const h = Buffer.alloc(44);
  const dataSize = dataLen * BYTES_PER_SAMPLE;
  h.write("RIFF", 0);
  h.writeUInt32LE(36 + dataSize, 4);
  h.write("WAVE", 8);
  h.write("fmt ", 12);
  h.writeUInt32LE(16, 16);
  h.writeUInt16LE(1, 20);
  h.writeUInt16LE(1, 22);
  h.writeUInt32LE(SAMPLE_RATE, 24);
  h.writeUInt32LE(SAMPLE_RATE * BYTES_PER_SAMPLE, 28);
  h.writeUInt16LE(BYTES_PER_SAMPLE, 32);
  h.writeUInt16LE(16, 34);
  h.write("data", 36);
  h.writeUInt32LE(dataSize, 40);
  return h;
}

function downsamplePcm(src: Buffer, srcRate: number): Buffer {
  if (srcRate <= SAMPLE_RATE) return src;
  const ratio = srcRate / SAMPLE_RATE;
  const outLen = Math.floor(src.length / 2 / ratio) * 2;
  const out = Buffer.alloc(outLen);
  for (let i = 0; i < outLen / 2; i++) {
    const srcIdx = Math.round(i * ratio);
    out.writeInt16LE(src.readInt16LE(srcIdx * 2), i * 2);
  }
  return out;
}

export async function transcribe(pcmBuffer: Buffer): Promise<string> {
  const wavHeader = createWavHeader(pcmBuffer.length / BYTES_PER_SAMPLE);
  const wavBuf = Buffer.concat([wavHeader, pcmBuffer]);
  return transcribeAudio(wavBuf, "audio/wav");
}

export async function transcribeAudio(
  audioBuffer: Buffer,
  mimeType: string
): Promise<string> {
  const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  const ext = mimeType.split("/")[1] || "wav";
  const tmpSrc = `/tmp/voice-${id}.${ext}`;
  const tmpWav = `/tmp/voice-${id}.wav`;
  const txtPath = `${tmpWav}.txt`;

  await fsPromises.writeFile(tmpSrc, audioBuffer);

  try {
    // Convert to WAV if needed (whisper.cpp supports wav, flac, mp3, ogg — not webm)
    if (ext !== "wav" && ext !== "flac" && ext !== "mp3" && ext !== "ogg") {
      execSync(
        `/tmp/ffmpeg -y -i ${tmpSrc} -ar 16000 -ac 1 -c:a pcm_s16le ${tmpWav} 2>/dev/null`
      );
    } else {
      await fsPromises.copyFile(tmpSrc, tmpWav);
    }

    execSync(
      `/private/tmp/whisper.cpp/build/bin/whisper-cli -m /tmp/whisper.cpp/models/ggml-tiny.en.bin -f ${tmpWav} -otxt -nt -np 2>/dev/null`
    );
    const text = await fsPromises.readFile(txtPath, "utf8");
    return text.trim();
  } catch {
    return "";
  } finally {
    fsPromises.unlink(tmpSrc).catch(() => {});
    fsPromises.unlink(tmpWav).catch(() => {});
    fsPromises.unlink(txtPath).catch(() => {});
  }
}

const conversationMemory = new Map<string, { role: string; content: string }[]>();
export const bookingStore = new Map<string, { name: string; refNo: string; model: string; variant: string; date: string; time: string }>();

function generateRefNo(): string {
  return String(100000 + Math.floor(Math.random() * 900000));
}

export async function askLLM(transcript: string, sessionId?: string): Promise<string> {
  const messages: { role: string; content: string }[] = [
    {
      role: "system",
      content:
        "You are Alex, a professional sales executive at RK Royal Engineers Pvt. Ltd., a Royal Enfield showroom in Hyderabad, India. You are on a phone call with a customer. Maintain a formal, courteous, and polished tone at all times. Speak with warmth but remain professional.\n\n" +
        "ROYAL ENFIELD 2026 LINEUP — Colors available per model:\n" +
        "350cc (349cc J-series):\n" +
        "- Hunter 350: Dapper White, Rebel Black, Dapper Grey, Dapper Green\n" +
        "- Classic 350: Stealth Black, Halcyon Green, Signals Marsh Grey, Chrome Bronze, Redditch Red, Dark, Emerald\n" +
        "- Bullet 350: Army Green, Black, White, Battalion Black, Military Silver\n" +
        "- Meteor 350: Fireball Yellow, Stellar Blue, Aurora Green, Supernova Brown\n" +
        "- Goan Classic 350: Goa Beach, Sea Foam, Jet Black\n" +
        "450cc (liquid-cooled):\n" +
        "- Scram 440: Matt Green, Blazing Black, Silly White\n" +
        "- Himalayan 450: Kaza Brown, Pine Green, Slate Poppy Blue, Summit White\n" +
        "- Guerrilla 450: Smoke, Plume, Blaze, Gold, Apex Black\n" +
        "650cc (648cc parallel-twin):\n" +
        "- Interceptor 650: Orange Crush, Canyon Red, Baker Express, Mark 2, Black Pearl\n" +
        "- Continental GT 650: British Racing Green, Dux Deluxe, Slipstream Blue, Apex Grey\n" +
        "- Super Meteor 650: Astral Green, Astral Black, Celestial Red\n" +
        "- Shotgun 650: Sheet Metal Grey, Plasma Green, Drill Black\n" +
        "- Bear 650: Two-Tone Grey, Wild White, Boardwalk Teal\n" +
        "- Classic 650: Chrome Bronze, Hotrod Red, Classic Black\n" +
        "- Bullet 650: Military Black, Force Silver\n" +
        "Electric: Flying Flea C6 — retro electric\n\n" +
        "SALES PROTOCOL — Follow in order:\n" +
        "1. Greet formally: \"Good morning/afternoon! This is Alex from RK Royal Engineers Pvt. Ltd., Hyderabad.\" Ask the customer's name.\n" +
        "2. Ask which model they're interested in.\n" +
        "3. When they mention a model, APPRECIATE enthusiastically. Ask about color/variant preference. If they ask about available colors, list the color options for that model.\n" +
        "4. Do not mention pricing unless asked. If asked, say pricing will be discussed by the sales team.\n" +
        "5. Ask if they'd like to book a TEST RIDE at our Hyderabad showroom.\n" +
        "6. If yes, ask for preferred DATE and TIME. Understand relative dates (\"next Wednesday\", \"tomorrow\") and times (\"10am\", \"evening\").\n" +
        "7. Confirm: \"Perfect! Your test ride for [model] is booked on [day], [date] at [time]. We look forward to welcoming you at our Hyderabad showroom!\"\n" +
        "Today is Friday, July 10, 2026. Keep responses under 3 sentences. Use polished, professional language. Never mention you are an AI.",
    },
  ];

  if (sessionId && conversationMemory.has(sessionId)) {
    const history = conversationMemory.get(sessionId)!;
    messages.push(...history);
  }

  messages.push({ role: "user", content: transcript });
  const res = await fetch(ZEN_API_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${ZEN_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: ZEN_MODEL,
      messages,
      max_tokens: 10000,
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Zen API error ${res.status}: ${err}`);
  }

  const data = await res.json();
  const content = data.choices?.[0]?.message?.content || "";
  const reasoning = data.choices?.[0]?.message?.reasoning_content || "";
  console.log(`[Zen] reasoning_tokens=${data.usage?.completion_tokens_details?.reasoning_tokens || "?"} content="${content.slice(0, 80)}"`);

  const reply = content || "[no response]";

  // Save to conversation memory
  if (sessionId) {
    if (!conversationMemory.has(sessionId)) {
      conversationMemory.set(sessionId, []);
    }
    const history = conversationMemory.get(sessionId)!;
    history.push({ role: "user", content: transcript });
    history.push({ role: "assistant", content: reply });
    // Keep last 20 messages to avoid unbounded growth
    if (history.length > 20) {
      conversationMemory.set(sessionId, history.slice(-20));
    }
  }

  // If booking is being set up, generate ref and add to context
  const existingBooking = sessionId ? bookingStore.get(sessionId) : undefined;
  if (!existingBooking && sessionId && transcript.toLowerCase().includes("book")) {
    const refNo = generateRefNo();
    bookingStore.set(sessionId, { name: "", refNo, model: "", variant: "", date: "", time: "" });
    // Add the ref number to the messages so Alex can say it
    messages.push({ role: "system", content: `Booking reference: ${refNo} (internal use only).` });
    // Re-fetch from API with updated messages
    const updatedRes = await fetch(ZEN_API_URL, {
      method: "POST",
      headers: { Authorization: `Bearer ${ZEN_API_KEY}`, "Content-Type": "application/json" },
      body: JSON.stringify({ model: ZEN_MODEL, messages, max_tokens: 10000 }),
    });
    if (updatedRes.ok) {
      const updatedData = await updatedRes.json();
      const updatedContent = updatedData.choices?.[0]?.message?.content || "";
      const updatedReply = updatedContent || "[no response]";
      const history = conversationMemory.get(sessionId);
      if (history) history[history.length - 1] = { role: "assistant", content: updatedReply };
      return updatedReply;
    }
  }

  // Try to extract booking info from the conversation
  if (sessionId && reply.toLowerCase().includes("test ride")) {
    const fullConvo = messages.slice(1).map(m => `${m.role}: ${m.content}`).join("\n");
    const nameMatch = fullConvo.match(/name is (\w+)/i) || fullConvo.match(/my name is (\w+)/i) || fullConvo.match(/calling from (\w+)/i);
    const modelMatch = fullConvo.match(/(hunter|classic|bullet|meteor|goan|scram|himalayan|guerrilla|interceptor|continental|super meteor|shotgun|bear|flying flea)\s*\d*/i);
    const dateMatch = fullConvo.match(/\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4})\b/i);
    const timeMatch = fullConvo.match(/(\d{1,2}(?::\d{2})?\s*(?:am|pm))/i);

    if (nameMatch || modelMatch) {
      bookingStore.set(sessionId, {
        name: nameMatch?.[1] || "",
        refNo: generateRefNo(),
        model: modelMatch?.[1] || "",
        variant: "",
        date: dateMatch?.[1] || "",
        time: timeMatch?.[1] || "",
      });
    }
  }

  return reply;
}

export async function elevenlabsTTS(text: string, format: string = "mp3_44100_128"): Promise<Buffer> {
  const res = await fetch(`${ELEVENLABS_API_URL}?output_format=${format}`, {
    method: "POST",
    headers: {
      "xi-api-key": ELEVENLABS_API_KEY!,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text,
      model_id: "eleven_multilingual_v2",
    }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`ElevenLabs error ${res.status}: ${err}`);
  }
  return Buffer.from(await res.arrayBuffer());
}

async function synthesize(text: string): Promise<Buffer | null> {
  if (process.env.TTS_ENABLED !== "true") return null;
  return elevenlabsTTS(text, "ulaw_8000");
}

export function attachMediaStreamHandler(server: Server) {
  const twilioWss = new WebSocketServer({ noServer: true });
  const demoWss = new WebSocketServer({ noServer: true });

  server.on("upgrade", (req, socket, head) => {
    const url = new URL(req.url ?? "/", `http://${req.headers.host}`);
    if (url.pathname === "/media-stream") {
      twilioWss.handleUpgrade(req, socket, head, (ws) => {
        twilioWss.emit("connection", ws, req);
      });
    } else if (url.pathname === "/voice-demo") {
      demoWss.handleUpgrade(req, socket, head, (ws) => {
        demoWss.emit("connection", ws, req);
      });
    } else {
      socket.destroy();
    }
  });

  demoWss.on("connection", (browserWs) => {
    console.log("[VoiceDemo] Browser connected");

    browserWs.on("message", async (raw) => {
      const msg = JSON.parse(raw.toString());

      if (msg.type === "audio-webm") {
        const audioBuf = Buffer.from(msg.data, "base64");
        console.log(`[VoiceDemo] Received audio: ${(audioBuf.length / 1024).toFixed(1)} KB`);
        try {
          const transcript = await transcribeAudio(audioBuf, "audio/webm");
          console.log(`[VoiceDemo] Transcript: "${transcript}"`);
          if (!transcript.trim()) {
            browserWs.send(JSON.stringify({ type: "result", transcript: "", reply: "" }));
            return;
          }
          const reply = await askLLM(transcript);
          console.log(`[VoiceDemo] Reply: "${reply}"`);
          browserWs.send(JSON.stringify({ type: "result", transcript, reply }));
        } catch (err: any) {
          console.error(`[VoiceDemo] Error:`, err.message);
          browserWs.send(JSON.stringify({ type: "error", message: err.message }));
        }
      }
    });

    browserWs.on("close", () => {
      console.log("[VoiceDemo] Browser disconnected");
    });
    browserWs.on("error", (err) => {
      console.error("[VoiceDemo] Browser error:", err.message);
    });
  });

  twilioWss.on("connection", (twilioWs) => {
    let streamSid: string | null = null;
    let audioChunks: Buffer[] = [];
    let silenceTimer: ReturnType<typeof setTimeout> | null = null;

    async function flushAudio() {
      if (audioChunks.length === 0) return;
      const pcmBuf = Buffer.concat(audioChunks);
      audioChunks = [];

      try {
        const transcript = await transcribe(pcmBuf);
        if (!transcript.trim()) return;
        console.log(`[User] ${transcript}`);

        const reply = await askLLM(transcript);
        console.log(`[Bot] ${reply}`);

        const ulawBuf = await synthesize(reply);
        if (!ulawBuf) return;
        const CHUNK_SIZE = 160;
        for (let i = 0; i < ulawBuf.length; i += CHUNK_SIZE) {
          if (twilioWs.readyState !== WebSocket.OPEN) break;
          const chunk = ulawBuf.subarray(i, i + CHUNK_SIZE);
          twilioWs.send(
            JSON.stringify({
              event: "media",
              streamSid,
              media: { payload: chunk.toString("base64") },
            })
          );
        }
      } catch (err) {
        console.error("[Pipeline] Error:", err);
      }
    }

    function resetSilenceTimer() {
      if (silenceTimer) clearTimeout(silenceTimer);
      silenceTimer = setTimeout(flushAudio, SILENCE_TIMEOUT_MS);
    }

    twilioWs.on("message", (raw) => {
      const msg = JSON.parse(raw.toString());

      switch (msg.event) {
        case "start":
          streamSid = msg.streamSid;
          audioChunks = [];
          break;

        case "media": {
          const ulaw = Buffer.from(msg.media.payload, "base64");
          const pcm = ulawToPcm(ulaw);
          audioChunks.push(pcm);
          resetSilenceTimer();
          break;
        }

        case "stop":
          if (silenceTimer) clearTimeout(silenceTimer);
          flushAudio();
          break;
      }
    });

    twilioWs.on("close", () => {
      if (silenceTimer) clearTimeout(silenceTimer);
    });

    twilioWs.on("error", () => {});
  });
}
