import "dotenv/config";
import express from "express";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";
import twilio from "twilio";
import VoiceResponse from "twilio/lib/twiml/VoiceResponse.js";
import { attachMediaStreamHandler, askLLM, transcribeAudio, bookingStore, elevenlabsTTS } from "./agent.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();
app.use(express.urlencoded({ extended: false }));
app.use(express.json({ limit: "10mb" }));
app.use(express.static(resolve(__dirname, "..")));

app.get("/", (_req, res) => res.redirect("/voice-demo.html"));

const PORT = process.env.PORT ? parseInt(process.env.PORT) : 8080;
const DOMAIN = process.env.DOMAIN;

const twilioClient =
  process.env.TWILIO_ACCOUNT_SID?.startsWith("AC")
    ? twilio(process.env.TWILIO_ACCOUNT_SID!, process.env.TWILIO_AUTH_TOKEN!)
    : null;

app.post("/call", async (req, res) => {
  if (!twilioClient) return res.status(400).json({ error: "Twilio not configured" });
  const { to } = req.body;
  if (!to) return res.status(400).json({ error: "Missing 'to' number" });

  const call = await twilioClient.calls.create({
    from: process.env.TWILIO_PHONE_NUMBER!,
    to,
    url: `https://${DOMAIN}/outbound-twiml`,
    statusCallback: `https://${DOMAIN}/call-status`,
    statusCallbackEvent: ["initiated", "ringing", "answered", "completed"],
  });

  res.json({ success: true, callSid: call.sid });
});

app.post("/outbound-twiml", (_req, res) => {
  const twiml = new VoiceResponse();
  const start = twiml.connect();
  start.stream({ url: `wss://${DOMAIN}/media-stream` });

  res.type("text/xml");
  res.send(twiml.toString());
});

app.post("/call-status", (req, res) => {
  console.log(`[Call ${req.body.CallStatus}] ${req.body.CallSid}`);
  res.sendStatus(200);
});

app.post("/demo", async (req, res) => {
  const { audio, mime, sessionId } = req.body;
  if (!audio) return res.status(400).json({ error: "Missing 'audio' (base64)" });

  try {
    const audioBuf = Buffer.from(audio, "base64");
    const mimeType = mime || "audio/wav";

    console.log(`[Demo] Audio received: ${(audioBuf.length / 1024).toFixed(1)} KB, type: ${mimeType} session=${sessionId || "none"}`);

    const transcript = await transcribeAudio(audioBuf, mimeType);
    console.log(`[Demo] Transcript: "${transcript}"`);

    if (!transcript.trim()) return res.json({ transcript: "", reply: "" });

    const reply = await askLLM(transcript, sessionId);
    console.log(`[Demo] Reply: "${reply}"`);

    let replyAudio: string | undefined;
    if (reply.trim() && process.env.ELEVENLABS_API_KEY && process.env.TTS_ENABLED === "true") {
      try {
        const audioBuf = await elevenlabsTTS(reply, "mp3_44100_128");
        replyAudio = audioBuf.toString("base64");
      } catch (err: any) {
        console.error(`[Demo] TTS error: ${err.message}`);
      }
    }

    const booking = sessionId ? bookingStore.get(sessionId) : undefined;
    res.json({
      transcript,
      reply,
      reply_audio: replyAudio,
      input_length: audioBuf.length,
      booking: booking && booking.name ? booking : undefined,
    });
  } catch (err: any) {
    console.error(`[Demo] Error:`, err.message);
    res.status(500).json({ error: err.message });
  }
});

app.post("/chat", async (req, res) => {
  const { message } = req.body;
  if (!message) return res.status(400).json({ error: "Missing 'message'" });

  try {
    const reply = await askLLM(message);
    console.log(`[Chat] "${message}" → "${reply}"`);
    res.json({ reply });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

const server = app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`DOMAIN=${DOMAIN}`);
});

attachMediaStreamHandler(server);
