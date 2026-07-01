import { settings } from "./config.js";

// The dataset is deliberately not linearly separable: class 0 is inside,
// class 1 is outside, so every straight cut must slice through both classes.
export function generateRings(random) {
  const points = [];
  const half = settings.pointCount / 2;

  for (let i = 0; i < settings.pointCount; i++) {
    const isOuterRing = i >= half;
    const angle = random.next() * Math.PI * 2;
    const baseRadius = isOuterRing ? 1.35 : 0.62;
    const radius = baseRadius + random.normal() * 0.08;

    points.push({
      x: Math.cos(angle) * radius + random.normal() * 0.025,
      y: Math.sin(angle) * radius + random.normal() * 0.025,
      label: isOuterRing ? 1 : 0
    });
  }

  return points;
}
