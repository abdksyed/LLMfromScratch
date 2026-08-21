(function () {
  const N = 12;

  const patterns = {
    dense: (i, j) => 0.15 + 0.85 * Math.exp(-Math.abs(i - j) / 4),
    causal: (i, j) => (j > i ? 0 : 0.12 + 0.88 * Math.exp(-(i - j) / 3.5)),
    window: (i, j) => {
      if (j > i) return 0;
      return i - j <= 3 ? 0.3 + 0.7 * Math.exp(-(i - j) / 2) : 0;
    },
    strided: (i, j) => {
      if (j > i) return 0;
      if (i - j <= 1 || j % 3 === 0 || i === j) return 0.85;
      return 0;
    },
    lsh: (i, j) => {
      if (j > i) return 0;
      const bucket = (x) => Math.floor(((x * 5 + 3) % N) / 4);
      return bucket(i) === bucket(j) || i - j <= 1 ? 0.8 : 0.02;
    },
    topk: (i, j) => {
      if (j > i) return 0;
      const score = ((i * 7 + j * 13) % 11) / 11;
      return score > 0.62 || i === j || j < 2 ? 0.15 + 0.85 * score : 0;
    },
    sink: (i, j) => {
      if (j > i) return 0;
      if (j === 0) return 0.95;
      return i - j <= 4 ? 0.35 + 0.55 * Math.exp(-(i - j) / 2) : 0;
    },
    linear: (i, j) => (j > i ? 0 : 0.08 + 0.35 * Math.exp(-(i - j) / 8)),
    rope: (i, j) => {
      if (j > i) return 0;
      const rel = i - j;
      return 0.12 + 0.88 * Math.abs(Math.cos(rel / 2.2)) * Math.exp(-rel / 7);
    },
    alibi: (i, j) => {
      if (j > i) return 0;
      return Math.max(0.05, 1 - (i - j) / (N * 1.15));
    },
    mqa: (i, j) => (j > i ? 0 : 0.2 + 0.6 * Math.exp(-(i - j) / 4)),
    sparse: (i, j) => {
      if (j > i) return 0;
      const local = i - j <= 2;
      const global = j % 4 === 0;
      return local || global ? 0.85 : 0;
    },
    nsa: (i, j) => {
      if (j > i) return 0;
      const compressed = Math.floor(j / 3) === Math.floor(i / 3);
      const selected = (j + i) % 5 === 0;
      const local = i - j <= 2;
      return compressed || selected || local ? 0.75 : 0.04;
    },
    flash: (i, j) => {
      if (j > i) return 0;
      const block = Math.floor(i / 4) === Math.floor(j / 4);
      return block ? 0.9 : 0.18 * Math.exp(-(i - j) / 6);
    },
  };

  function color(v, era) {
    const palettes = {
      exact: [37, 99, 235],    // blue
      memory: [13, 148, 136],  // teal
      length: [180, 83, 9],    // amber / warm brown
      hybrid: [109, 40, 217],  // violet
    };
    const [r, g, b] = palettes[era] || palettes.exact;
    const a = 0.15 + v * 0.85;
    return `rgba(${r},${g},${b},${a.toFixed(3)})`;
  }

  function draw(canvas) {
    const ctx = canvas.getContext("2d");
    const type = canvas.dataset.pattern;
    const era = canvas.dataset.era || "exact";
    const fn = patterns[type] || patterns.causal;
    const dpr = window.devicePixelRatio || 1;
    const size = 160;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = "#f8fafc";
    ctx.fillRect(0, 0, size, size);
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, size - 1, size - 1);
    const pad = 8;
    const cell = (size - pad * 2) / N;
    const gap = 1.4;
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const v = Math.max(0, Math.min(1, fn(i, j)));
        ctx.fillStyle = v < 0.03 ? "#ffffff" : color(v, era);
        ctx.fillRect(pad + j * cell, pad + i * cell, cell - gap, cell - gap);
      }
    }
  }

  document.querySelectorAll("canvas.attn").forEach(draw);

  const links = [...document.querySelectorAll(".rail a")];
  const mechs = [...document.querySelectorAll(".mech")];
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + e.target.id));
      });
    },
    { rootMargin: "-40% 0px -50% 0px", threshold: 0.01 }
  );
  mechs.forEach((m) => io.observe(m));

  const buttons = document.querySelectorAll(".filters button");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const era = btn.dataset.era;
      buttons.forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
      mechs.forEach((m) => {
        if (era === "all") m.classList.remove("hidden");
        else m.classList.toggle("hidden", m.dataset.era !== era);
      });
      links.forEach((a) => {
        const target = document.querySelector(a.getAttribute("href"));
        if (!target) return;
        a.style.display = era === "all" || target.dataset.era === era ? "" : "none";
      });
    });
  });
})();
