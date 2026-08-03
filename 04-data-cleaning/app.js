const progress = document.querySelector(".reading-progress span");
const navLinks = [...document.querySelectorAll(".section-nav a")];
const sections = navLinks
  .map((link) => document.querySelector(link.getAttribute("href")))
  .filter(Boolean);

function updateProgress() {
  if (!progress) return;
  const available = document.documentElement.scrollHeight - window.innerHeight;
  const value = available > 0 ? (window.scrollY / available) * 100 : 0;
  progress.style.width = `${Math.min(100, Math.max(0, value))}%`;
}

const observer = new IntersectionObserver(
  (entries) => {
    const visible = entries
      .filter((entry) => entry.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

    if (!visible) return;

    navLinks.forEach((link) => {
      const active = link.getAttribute("href") === `#${visible.target.id}`;
      if (active) link.setAttribute("aria-current", "true");
      else link.removeAttribute("aria-current");
    });
  },
  { rootMargin: "-25% 0px -60%", threshold: [0, 0.2, 0.5] },
);

sections.forEach((section) => observer.observe(section));
window.addEventListener("scroll", updateProgress, { passive: true });
updateProgress();

/* ── Strategy cards: expand / collapse ─────────────────────────────────────── */
const strategyList = document.querySelector(".strategy-list");

if (strategyList) {
  strategyList.addEventListener("click", (event) => {
    const trigger = event.target.closest(".strategy-head");
    if (!trigger) return;
    const card = trigger.closest(".strategy");
    const open = card.getAttribute("data-open") === "true";
    card.setAttribute("data-open", open ? "false" : "true");
    trigger.setAttribute("aria-expanded", open ? "false" : "true");
  });

  const expandAll = document.querySelector("[data-expand-all]");
  if (expandAll) {
    expandAll.addEventListener("click", () => {
      const cards = [...strategyList.querySelectorAll(".strategy")];
      const shouldOpen = cards.some((c) => c.getAttribute("data-open") !== "true");
      cards.forEach((card) => {
        card.setAttribute("data-open", shouldOpen ? "true" : "false");
        card
          .querySelector(".strategy-head")
          .setAttribute("aria-expanded", shouldOpen ? "true" : "false");
      });
      expandAll.textContent = shouldOpen ? "Collapse all" : "Expand all";
    });
  }
}
