import { settings } from "./config.js";

function plotToPixel(value, size) {
  const span = settings.plotMax - settings.plotMin;
  return ((value - settings.plotMin) / span) * size;
}

export function drawPlot(canvas, points, predict) {
  const ctx = canvas.getContext("2d");
  const size = canvas.width;
  const span = settings.plotMax - settings.plotMin;

  ctx.clearRect(0, 0, size, size);

  // Paint the background by asking the model what class each grid cell is.
  // The color switch at 50% probability is the visible decision boundary.
  for (let py = 0; py < size; py += settings.gridStep) {
    for (let px = 0; px < size; px += settings.gridStep) {
      const x = settings.plotMin + (px / size) * span;
      const y = settings.plotMax - (py / size) * span;
      const probability = predict(x, y);
      ctx.fillStyle = probability >= 0.5
        ? "rgba(227, 79, 79, 0.18)"
        : "rgba(36, 107, 254, 0.18)";
      ctx.fillRect(px, py, settings.gridStep + 1, settings.gridStep + 1);
    }
  }

  ctx.strokeStyle = "rgba(23, 32, 38, 0.2)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(size / 2, 0);
  ctx.lineTo(size / 2, size);
  ctx.moveTo(0, size / 2);
  ctx.lineTo(size, size / 2);
  ctx.stroke();

  for (const point of points) {
    const px = plotToPixel(point.x, size);
    const py = size - plotToPixel(point.y, size);

    ctx.beginPath();
    ctx.arc(px, py, 4.2, 0, Math.PI * 2);
    ctx.fillStyle = point.label === 1 ? "#e34f4f" : "#246bfe";
    ctx.fill();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}
