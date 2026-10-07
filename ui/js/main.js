// Điểm vào: sidebar các bước, router theo hash (#/chunking, #/embedding, …), nạp dữ liệu dùng chung.
import { get } from "./api.js";
import chunkingLesson from "./lessons/chunking.js";
import embeddingLesson from "./lessons/embedding.js";
import indexLesson from "./lessons/index-store.js";
import transformLesson from "./lessons/query-transform.js";
import retrievalLesson from "./lessons/retrieval.js";
import rerankLesson from "./lessons/rerank.js";
import evaluateLesson from "./lessons/evaluate.js";
import pipelineLesson from "./lessons/pipeline.js";
import overview from "./steps/overview.js";
import { lessonStep } from "./lesson/shell.js";
import { $, esc } from "./util.js";

const STEPS = [overview, ...[chunkingLesson, embeddingLesson, indexLesson, transformLesson, retrievalLesson, rerankLesson, evaluateLesson, pipelineLesson].map(lessonStep)];
const GROUPS = { 0: "", A: "A · Chuẩn bị dữ liệu (offline)", B: "B · Xử lý mỗi query (online)", C: "C · Đo lường và vận hành" };

function renderNav() {
  $("#steps").innerHTML = Object.entries(GROUPS).map(([g, label]) => (label ? `<div class="navgroup">${esc(label)}</div>` : "") +
    STEPS.filter((s) => s.group === g).map((s) =>
      `<a class="navitem" href="#/${s.id}" data-id="${s.id}"><span class="step">${s.num}</span><span>${esc(s.title)}</span></a>`).join("")).join("");
}

/** Tách "#/rerank?q=abc" thành id bước + tham số (gộp cả ?query của URL chính). */
function parseRoute() {
  const [path, hashQuery = ""] = location.hash.replace(/^#\//, "").split("?");
  const params = new URLSearchParams(location.search);
  new URLSearchParams(hashQuery).forEach((v, k) => params.set(k, v));
  return { id: path, params };
}

function currentStep(params) {
  const { id } = parseRoute();
  if (!id && params.get("tab") === "eval") return STEPS.find((s) => s.id === "evaluate");  // link cũ: ?tab=eval
  return STEPS.find((s) => s.id === id) ?? STEPS[0];
}

async function init() {
  renderNav();
  const view = $("#view"), status = $("#status");
  let ctx;
  try {
    const [health, meta, space] = await Promise.all([get("/health"), get("/api/meta"), get("/api/space")]);
    status.textContent = `kết nối OK · ${health.chunks} chunk`;
    status.classList.add("ok");
    ctx = { meta, space, params: new URLSearchParams(location.search) };
  } catch (err) {
    status.textContent = "không kết nối được backend";
    status.classList.add("fail");
    view.innerHTML = `<div class="error">Backend/DB chưa sẵn sàng: ${esc(err.message)}. Chạy <code>docker compose up -d</code> rồi tải lại trang.</div>`;
    return;
  }

  async function route() {
    ctx.params = parseRoute().params;
    const step = currentStep(ctx.params);
    document.querySelectorAll(".navitem").forEach((a) => a.classList.toggle("active", a.dataset.id === step.id));
    const root = document.createElement("div");  // mỗi lần vào bước dùng container mới: tác vụ async của bước cũ không ghi đè trang mới
    view.replaceChildren(root);
    document.title = `${step.num}. ${step.title} · Embedding & Retrieval Lab`;
    try { await step.mount(root, ctx); }
    catch (err) { root.insertAdjacentHTML("beforeend", `<div class="error">Lỗi dựng bước "${esc(step.title)}": ${esc(err.message)}</div>`); console.error(err); }
    window.scrollTo(0, 0);
  }
  window.addEventListener("hashchange", route);
  route();
}

init();
