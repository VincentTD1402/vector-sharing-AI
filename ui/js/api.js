// Gọi backend FastAPI.
async function request(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch { /* giữ statusText */ }
    throw new Error(`${response.status}: ${detail}`);
  }
  return response.json();
}

export const get = (path) => request(path);
export const post = (path, body = {}) =>
  request(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

/** Đọc NDJSON streaming: gọi onEvent cho từng dòng ngay khi server gửi (để cập nhật tiến độ). */
export async function streamNdjson(path, onEvent) {
  const response = await fetch(path);
  if (!response.ok || !response.body) throw new Error(`${response.status}: ${response.statusText}`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newline;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (line) onEvent(JSON.parse(line));
    }
  }
}
