// Khung một BÀI HỌC: ① lý thuyết → ② tình huống/lỗi trong slide → ③ chạy từng bước → ④ chẩn đoán, sửa, so sánh.
import { stepHeader } from "../components.js";
import { $, esc } from "../util.js";
import { callout } from "./render.js";
import { mountPlayer } from "./player.js";

const TAGS = { problem: "Vấn đề", error: "Lỗi", fix: "Giải pháp", observe: "Quan sát", free: "Tự thử" };

function sectionHead(n, title, sub = "") {
  return `<div class="sechead"><span class="step">${n}</span><h2>${esc(title)}</h2>${sub ? `<span class="hint">${esc(sub)}</span>` : ""}</div>`;
}

function theoryHtml(t) {
  return `<div class="theorygrid">
    <div><h3>Khái niệm</h3><div class="prose">${t.what}</div></div>
    <div><h3>Vì sao quan trọng</h3><div class="prose">${t.why}</div></div>
    <div><h3>Cần nhớ</h3><ul class="prose">${t.points.map((p) => `<li>${p}</li>`).join("")}</ul></div></div>
    ${t.visual ? `<div class="theoryvisual">${t.visual}</div>` : ""}`;
}

function fieldHtml(f, value) {
  const v = value ?? f.default ?? "";
  const label = `<span>${esc(f.label)}${f.type === "range" ? `: <output>${esc(v)}</output>${f.unit ? " " + esc(f.unit) : ""}` : ""}</span>`;
  const controls = {
    text: () => `<input type="text" name="${f.name}" value="${esc(v)}" maxlength="500">`,
    textarea: () => `<textarea name="${f.name}" rows="${f.rows ?? 4}">${esc(v)}</textarea>`,
    number: () => `<input type="number" name="${f.name}" value="${esc(v)}" min="${f.min ?? ""}" max="${f.max ?? ""}">`,
    range: () => `<input type="range" name="${f.name}" min="${f.min}" max="${f.max}" step="${f.step ?? 1}" value="${esc(v)}">`,
    check: () => `<input type="checkbox" name="${f.name}" ${v ? "checked" : ""}>`,
    select: () => `<select name="${f.name}">${f.options.map((o) => `<option value="${esc(o.value)}" ${String(o.value) === String(v) ? "selected" : ""}>${esc(o.label)}</option>`).join("")}</select>`,
  };
  const ctl = controls[f.type]();
  return `<label class="field ${f.type === "check" ? "check" : ""} ${f.wide ? "wide" : ""}">${f.type === "check" ? ctl + label : label + ctl}</label>`;
}

function readValues(form, fields) {
  const out = {};
  for (const f of fields) {
    const el = form.elements[f.name];
    if (!el) continue;
    out[f.name] = f.type === "check" ? el.checked : (f.type === "range" || f.type === "number") ? Number(el.value) : el.value;
  }
  return out;
}

function factCell(v) {
  if (v && typeof v === "object") return `<td class="fact ${v.level ?? ""}">${v.v}</td>`;
  return `<td>${v ?? "–"}</td>`;
}

/** Bọc định nghĩa bài học thành một 'step' cho router. */
export const lessonStep = (def) => ({ ...def, mount(root, ctx) { mountLesson(root, def, ctx); } });

export function mountLesson(root, def, ctx) {
  root.innerHTML = `${stepHeader(def)}
    <section class="card lessoncard" id="s-theory">${sectionHead(1, "Lý thuyết", `Slide ${def.slides} · đọc ~1 phút`)}${theoryHtml(def.theory)}</section>
    <section class="card lessoncard" id="s-cases">${sectionHead(2, "Chọn một tình huống trong slide", "Mỗi thẻ là một vấn đề/lỗi/giải pháp thật; bấm để nạp dữ liệu, CHƯA chạy gì")}
      <div class="cases">${def.cases.map((c, i) => `<button type="button" class="casecard ${c.tag}" data-i="${i}">
        <span class="tag ${c.tag}">${TAGS[c.tag]}</span><b>${esc(c.title)}</b><span class="story">${esc(c.story)}</span>
        ${c.slide ? `<span class="slidetag">slide ${esc(c.slide)}</span>` : ""}</button>`).join("")}</div>
      <div id="casedetail"></div></section>
    <section class="card lessoncard" id="s-run" hidden>${sectionHead(3, "Chạy từng bước", "Thanh luồng bên dưới là toàn bộ quy trình; bấm ▶ để mở từng bước")}<div id="runarea"></div></section>
    <section class="card lessoncard" id="s-verdict" hidden></section>`;

  const detail = $("#casedetail", root), runCard = $("#s-run", root), runArea = $("#runarea", root), verdictCard = $("#s-verdict", root);
  let current = null, runs = [], player = null;

  function selectCase(i) {
    current = def.cases[i]; runs = []; player?.stop(); player = null;
    root.querySelectorAll(".casecard").forEach((b) => b.classList.toggle("sel", Number(b.dataset.i) === i));
    runCard.hidden = true; verdictCard.hidden = true; verdictCard.innerHTML = ""; runArea.innerHTML = "";
    detail.innerHTML = `<div class="casebox ${current.tag}">
      <div class="casetext"><h3>${esc(current.title)}</h3><p><b>Slide nói:</b> ${esc(current.story)}</p>
        ${current.expect ? `<p><b>Bạn sẽ thấy:</b> ${current.expect}</p>` : ""}</div>
      ${current.custom ? `<button type="button" class="primary" id="go">Mở thí nghiệm ▶</button>` : `
      <form id="caseform" class="fields">${def.fields.filter((f) => !f.hideFor?.includes(current.id) && (!f.showFor || f.showFor.includes(current.id))).map((f) => fieldHtml(f, current.values?.[f.name])).join("")}</form>
      <div class="row"><button type="button" class="primary" id="go">▶ Bắt đầu chạy</button><span class="hint">Bạn có thể sửa dữ liệu trên trước khi chạy.</span></div>`}</div>`;
    const form = $("#caseform", root);
    form?.addEventListener("input", (e) => { const o = e.target.closest("label")?.querySelector("output"); if (o) o.textContent = e.target.value; });
    $("#go", root).onclick = () => (current.custom ? openCustom() : start(readValues(form, def.fields), "Gốc"));
    detail.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function openCustom() {
    runCard.hidden = false; verdictCard.hidden = true;
    runArea.innerHTML = "<div id=\"customarea\"></div>";
    current.custom($("#customarea", root), ctx);
    runCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function start(values, label) {
    const go = $("#go", root);
    go.disabled = true; go.textContent = "Đang chạy trên backend…";
    runCard.hidden = false; verdictCard.hidden = true;
    runArea.innerHTML = `<div class="emptystate">Đang chạy trên backend…</div>`;
    try {
      const result = await def.run(values, ctx, current);
      runs.push({ label, values, result });
      player?.stop();
      player = mountPlayer(runArea, result.steps, { onDone: () => showVerdict(result) });
      runArea.addEventListener("player-reset", () => { verdictCard.hidden = true; }, { once: true });
      runCard.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      runArea.innerHTML = `<div class="error">Không chạy được: ${esc(err.message)}</div>`;
    } finally { go.disabled = false; go.textContent = "▶ Chạy lại với dữ liệu trên"; }
  }

  function showVerdict(result) {
    const d = result.diagnosis;
    const fixes = [...(current.fixes ?? []), ...(result.fixes ?? [])];
    const facts = [...new Set(runs.flatMap((r) => Object.keys(r.result.summary?.facts ?? {})))];
    verdictCard.hidden = false;
    verdictCard.innerHTML = `${sectionHead(4, "Chẩn đoán và bài học")}
      ${callout(d.level, `<b>${d.title}</b>${d.html ? `<div>${d.html}</div>` : ""}`)}
      ${result.lessons?.length ? `<h3>Rút ra</h3><ul class="prose">${result.lessons.map((l) => `<li>${l}</li>`).join("")}</ul>` : ""}
      ${fixes.length ? `<h3>Thử cách sửa (chạy lại ngay, rồi so sánh)</h3><div class="fixes">${fixes.map((f, i) =>
        `<button type="button" data-fix="${i}">${esc(f.label)}</button>`).join("")}</div>` : ""}
      ${runs.length > 1 ? `<h3>So sánh các lần chạy</h3><div class="tablewrap"><table class="runs"><thead><tr><th>Lần chạy</th>${facts.map((k) => `<th>${esc(k)}</th>`).join("")}</tr></thead><tbody>${
        runs.map((r) => `<tr><td><b>${esc(r.label)}</b><div class="hint">${r.result.summary?.label ?? ""}</div></td>${facts.map((k) => factCell(r.result.summary?.facts?.[k])).join("")}</tr>`).join("")}</tbody></table></div>` : ""}`;
    verdictCard.querySelectorAll("[data-fix]").forEach((b) => {
      b.onclick = () => {
        const fix = fixes[Number(b.dataset.fix)];
        if (fix.href) { location.hash = fix.href.replace(/^#/, ""); return; }
        const form = $("#caseform", root);
        for (const [k, v] of Object.entries(fix.patch)) {
          const el = form.elements[k];
          if (!el) continue;
          if (el.type === "checkbox") el.checked = !!v; else el.value = v;
          const out = el.closest("label")?.querySelector("output"); if (out) out.textContent = v;
        }
        start(readValues(form, def.fields), `Sau khi sửa: ${fix.label}`);
      };
    });
    verdictCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  root.querySelector(".cases").addEventListener("click", (e) => {
    const card = e.target.closest(".casecard");
    if (card) selectCase(Number(card.dataset.i));
  });
  if (ctx.params.get("case") != null) selectCase(Number(ctx.params.get("case")));
}
