/* Scan the World — test app v3. Raw PCM capture processor.
   Posts every render quantum's mono input samples back to the page
   with the quantum's exact starting frame (AudioWorkletGlobalScope
   currentFrame), so the page can place the recording on the audio
   context's sample timeline without guessing. Output is silence:
   the node exists only to observe the input. */
class StwPcmCapture extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    const ch = input && input[0];
    if (ch && ch.length) {
      this.port.postMessage({ frame: currentFrame, samples: ch.slice(0) });
    }
    return true;
  }
}
registerProcessor("stw-pcm-capture", StwPcmCapture);
