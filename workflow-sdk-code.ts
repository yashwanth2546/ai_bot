import { workflow, node, trigger, expr } from '@n8n/workflow-sdk';

const manualTrigger = trigger({
  type: 'n8n-nodes-base.manualTrigger',
  version: 1,
  config: { name: 'Manual Trigger', position: [240, 300] }
});

const readAudioFile = node({
  type: 'n8n-nodes-base.readBinaryFiles',
  version: 1,
  config: { name: 'Read Audio File', parameters: { filePath: '/Users/you/Desktop/test-audio.wav' }, position: [460, 300] }
});

const prepareJsonPayload = node({
  type: 'n8n-nodes-base.code',
  version: 2,
  config: {
    name: 'Prepare JSON Payload',
    parameters: {
      mode: 'runOnceForAllItems',
      language: 'javaScript',
      jsCode: 'const binary = $input.first().binary.data;\nreturn [{\n  json: {\n    audio: binary.data,\n    mime: binary.mimeType\n  }\n}];'
    },
    position: [680, 300]
  }
});

const postDemo = node({
  type: 'n8n-nodes-base.httpRequest',
  version: 4.4,
  config: {
    name: 'POST /demo',
    parameters: {
      url: 'http://localhost:8080/demo',
      method: 'POST',
      sendBody: true,
      contentType: 'json',
      specifyBody: 'json',
      jsonBody: expr('{{ $json }}')
    },
    position: [900, 300]
  }
});

export default workflow('ai-bot-pipeline', 'AI Bot - Full Pipeline Demo')
  .add(manualTrigger)
  .to(readAudioFile)
  .to(prepareJsonPayload)
  .to(postDemo);
