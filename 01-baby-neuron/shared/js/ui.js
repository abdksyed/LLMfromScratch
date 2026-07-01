export function $(selector) {
  return document.querySelector(selector);
}

export function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}

export function epochsPerFrame(totalEpochs, targetFrames = 120) {
  return Math.max(1, Math.ceil(totalEpochs / targetFrames));
}

export function bindEpochSlider(input, label) {
  label.textContent = input.value;
  input.addEventListener("input", () => {
    label.textContent = input.value;
  });
}

export function nextFrame() {
  return new Promise(requestAnimationFrame);
}
