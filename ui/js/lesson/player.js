// Trình phát các bước: thanh luồng (thấy cả quy trình ngay từ đầu) + thẻ từng bước mở ra lần lượt.
import { $, esc } from "../util.js";

const RUNNING_MS = 650;   // thời gian "đang chạy" mang tính trình diễn; thời gian THẬT của mỗi bước hiện ở huy hiệu ms
const AUTO_GAP_MS = 1200;

/**
 * steps: [{title, explain, ms?, html, summary?, warn?, after?(el)}]
 * Trả về {stop()}. onDone() được gọi một lần khi bước cuối đã hiện.
 */
export function mountPlayer(container, steps, { onDone } = {}) {
  const root = document.createElement("div");  // gốc MỚI mỗi lần dựng: listener của lần trước biến mất cùng nó
  root.className = "player";
  container.replaceChildren(root);
  root.innerHTML = `<div class="rail" aria-label="Các bước">${steps.map((s, i) =>
      `${i ? '<i class="railarrow">→</i>' : ""}<button type="button" class="node pending" data-i="${i}"><span class="n">${i + 1}</span><b>${esc(s.title)}</b></button>`).join("")}</div>
    <div class="controls">
      <button type="button" class="primary" data-act="next">▶ Bước tiếp</button>
      <button type="button" data-act="all">⏩ Chạy hết</button>
      <label class="check"><input type="checkbox" data-act="auto"> Tự động phát</label>
      <button type="button" data-act="reset">↺ Làm lại</button>
      <span class="counter"></span></div>
    <div class="stepcards"></div>`;
  const rail = root.querySelector(".rail"), cards = root.querySelector(".stepcards");
  const counter = root.querySelector(".counter"), next = root.querySelector('[data-act="next"]');
  let shown = 0, busy = false, timer = null, stopped = false, doneFired = false;

  const nodeOf = (i) => rail.querySelector(`.node[data-i="${i}"]`);
  const setCounter = () => { counter.textContent = `Bước ${shown}/${steps.length}`; };

  function collapseOthers(keep) {
    cards.querySelectorAll(".stepcard").forEach((c) => { if (Number(c.dataset.i) !== keep) c.classList.remove("open"); });
  }

  async function reveal() {
    if (busy || shown >= steps.length || stopped) return;
    busy = true; next.disabled = true;
    const i = shown, s = steps[i];
    nodeOf(i).className = "node running";
    const card = document.createElement("article");
    card.className = "stepcard running open"; card.dataset.i = i;
    card.innerHTML = `<header><span class="n">${i + 1}</span><h3>${esc(s.title)}</h3><span class="spin" aria-hidden="true"></span><span class="status">Đang chạy…</span></header>
      <div class="body"><p class="explain">${s.explain ?? ""}</p></div>`;
    collapseOthers(i);
    cards.appendChild(card);
    card.scrollIntoView({ behavior: "smooth", block: "center" });
    await new Promise((r) => setTimeout(r, RUNNING_MS));
    if (stopped) return;
    card.className = "stepcard done open";
    card.innerHTML = `<header><span class="n">${i + 1}</span><h3>${esc(s.title)}</h3>
        ${s.ms != null ? `<span class="msbadge" title="Thời gian thật đo ở backend">${esc(s.ms)}</span>` : ""}
        <span class="sum">${s.summary ?? ""}</span><button type="button" class="tog" aria-label="Thu gọn/mở rộng">▾</button></header>
      <div class="body"><p class="explain">${s.explain ?? ""}</p><div class="out">${s.html ?? ""}</div>${s.warn ? `<div class="callout warn">${s.warn}</div>` : ""}</div>`;
    s.after?.(card.querySelector(".out"));
    nodeOf(i).className = "node done";
    shown++; busy = false; setCounter();
    if (shown >= steps.length) {
      next.disabled = true;
      if (!doneFired) { doneFired = true; onDone?.(); }
    } else {
      next.disabled = false;
      if (root.querySelector('[data-act="auto"]').checked) timer = setTimeout(reveal, AUTO_GAP_MS);
    }
  }

  async function revealAll() {
    while (shown < steps.length && !stopped) { await reveal(); }
  }

  root.addEventListener("click", (e) => {
    const act = e.target.closest("[data-act]")?.dataset.act;
    if (act === "next") reveal();
    else if (act === "all") revealAll();
    else if (act === "reset") { clearTimeout(timer); stopped = true; mountPlayer(container, steps, { onDone }); container.dispatchEvent(new CustomEvent("player-reset")); }
    const node = e.target.closest(".node");
    if (node && node.classList.contains("done")) {
      const card = cards.querySelector(`.stepcard[data-i="${node.dataset.i}"]`);
      collapseOthers(Number(node.dataset.i)); card.classList.add("open"); card.scrollIntoView({ behavior: "smooth", block: "center" });
    }
    const tog = e.target.closest(".tog");
    if (tog) tog.closest(".stepcard").classList.toggle("open");
  });
  root.querySelector('[data-act="auto"]').addEventListener("change", (e) => {
    if (e.target.checked && !busy) reveal(); else if (!e.target.checked) clearTimeout(timer);
  });
  setCounter();
  return { stop() { stopped = true; clearTimeout(timer); }, revealAll };
}
