export class RandomSource {
  constructor(seed = 7) {
    this.seed = seed;
  }

  bump(amount = 101) {
    this.seed = (this.seed + amount) >>> 0;
  }

  // Tiny seeded random generator. Reproducible randomness makes the demo easier
  // to explain: the same seed creates the same rings and starting weights.
  next() {
    this.seed = (this.seed * 1664525 + 1013904223) >>> 0;
    return this.seed / 4294967296;
  }

  // Box-Muller transform: turns two uniform random numbers into a bell-curve
  // random number, useful for noisy rings and small random model weights.
  normal() {
    const a = Math.max(this.next(), 1e-8);
    const b = this.next();
    return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b);
  }
}
