const conversation = document.getElementById("conversation");
const messages = document.getElementById("messages");
const welcome = document.getElementById("welcome");
const form = document.getElementById("chatForm");
const input = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const sendIcon = document.getElementById("sendIcon");
const modelLabel = document.getElementById("modelLabel");
const sidebar = document.getElementById("sidebar");
const sidebarOverlay = document.getElementById("sidebarOverlay");
const themeButton = document.getElementById("themeButton");
const themeIcon = document.getElementById("themeIcon");
const themeLabel = document.getElementById("themeLabel");
const knowledgeModal = document.getElementById("knowledgeModal");
const knowledgeForm = document.getElementById("knowledgeForm");
const knowledgeMessage = document.getElementById("knowledgeMessage");
const knowledgeSubmit = document.getElementById("knowledgeSubmit");

let activeController = null;
let isGenerating = false;

function scrollToBottom() {
  requestAnimationFrame(() => {
    conversation.scrollTop = conversation.scrollHeight;
  });
}

function resizeInput() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
}

function setGenerating(value) {
  isGenerating = value;
  input.disabled = value;
  sendButton.disabled = false;
  sendButton.classList.toggle("stop", value);
  sendIcon.textContent = value ? "■" : "↑";
  sendButton.setAttribute("aria-label", value ? "停止生成" : "发送");
}

function closeSidebar() {
  sidebar.classList.remove("open");
  sidebarOverlay.classList.remove("visible");
}

function openKnowledgeModal() {
  knowledgeMessage.textContent = "";
  knowledgeMessage.className = "form-message";
  knowledgeModal.hidden = false;
  document.body.style.overflow = "hidden";
  knowledgeForm.elements.allusion_name.focus();
  closeSidebar();
}

function closeKnowledgeModal() {
  knowledgeModal.hidden = true;
  document.body.style.overflow = "";
}

function addUserMessage(text) {
  const article = document.createElement("article");
  article.className = "message user";

  const content = document.createElement("div");
  content.className = "message-content";
  const body = document.createElement("div");
  body.className = "message-text";
  body.textContent = text;
  content.appendChild(body);
  article.appendChild(content);
  messages.appendChild(article);
}

function addAssistantMessage() {
  const article = document.createElement("article");
  article.className = "message assistant";

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = "识";

  const content = document.createElement("div");
  content.className = "message-content";
  const status = document.createElement("div");
  status.className = "message-status";
  status.innerHTML = '<span class="thinking-dot"></span><span>正在准备解析…</span>';
  const selection = document.createElement("div");
  selection.hidden = true;
  const body = document.createElement("div");
  body.className = "message-text";
  const actions = document.createElement("div");
  actions.className = "message-actions";
  actions.hidden = true;

  const copyButton = document.createElement("button");
  copyButton.className = "copy-button";
  copyButton.type = "button";
  copyButton.textContent = "复制回答";
  copyButton.addEventListener("click", async () => {
    await navigator.clipboard.writeText(body.textContent);
    copyButton.textContent = "已复制";
    window.setTimeout(() => { copyButton.textContent = "复制回答"; }, 1200);
  });
  actions.appendChild(copyButton);

  content.append(status, selection, body, actions);
  article.append(avatar, content);
  messages.appendChild(article);
  return {article, status, selection, body, actions};
}

function showStatus(target, message) {
  target.hidden = false;
  target.innerHTML = "";
  const dot = document.createElement("span");
  dot.className = "thinking-dot";
  const label = document.createElement("span");
  label.textContent = message;
  target.append(dot, label);
}

function showSelection(target, event) {
  target.hidden = false;
  target.className = "selection-card";
  target.innerHTML = "";

  const label = document.createElement("span");
  label.textContent = "模型选择";
  const name = document.createElement("span");
  name.className = "selection-name";
  name.textContent = event.allusion_name;
  const score = document.createElement("span");
  score.className = "selection-score";
  score.textContent = `相似度 ${Number(event.score).toFixed(3)}`;
  target.append(label, name, score);
}

function parseEventBlock(block) {
  const dataLines = block
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart());
  if (!dataLines.length) {
    return null;
  }
  return JSON.parse(dataLines.join("\n"));
}

async function submitMessage(text) {
  welcome.hidden = true;
  addUserMessage(text);
  const assistant = addAssistantMessage();
  scrollToBottom();
  setGenerating(true);
  activeController = new AbortController();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({message: text}),
      signal: activeController.signal,
    });
    if (!response.ok || !response.body) {
      throw new Error(`接口返回 ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const {value, done} = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
      const blocks = buffer.split("\n\n");
      buffer = blocks.pop() || "";

      for (const block of blocks) {
        const event = parseEventBlock(block);
        if (!event) continue;

        if (event.type === "status") {
          showStatus(assistant.status, event.message);
        } else if (event.type === "selection") {
          showSelection(assistant.selection, event);
          showStatus(assistant.status, "正在生成典故解析…");
        } else if (event.type === "delta") {
          assistant.status.hidden = true;
          assistant.body.textContent += event.content;
        } else if (event.type === "refusal" || event.type === "error") {
          assistant.status.hidden = true;
          assistant.body.textContent = event.message;
        } else if (event.type === "done") {
          assistant.status.hidden = true;
        }
        scrollToBottom();
      }

      if (done) break;
    }
  } catch (error) {
    assistant.status.hidden = true;
    assistant.body.textContent = error.name === "AbortError"
      ? "已停止本次生成。"
      : `连接本地服务失败：${error.message}`;
  } finally {
    assistant.status.hidden = true;
    assistant.actions.hidden = !assistant.body.textContent;
    activeController = null;
    setGenerating(false);
    input.disabled = false;
    input.focus();
    scrollToBottom();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (isGenerating) {
    activeController?.abort();
    return;
  }

  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  resizeInput();
  submitMessage(text);
});

input.addEventListener("input", resizeInput);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.prompt;
    resizeInput();
    input.focus();
    closeSidebar();
  });
});

document.getElementById("newChatButton").addEventListener("click", () => {
  if (isGenerating) activeController?.abort();
  messages.innerHTML = "";
  welcome.hidden = false;
  input.value = "";
  resizeInput();
  closeSidebar();
  input.focus();
});

document.getElementById("knowledgeButton").addEventListener("click", openKnowledgeModal);
document.getElementById("knowledgeClose").addEventListener("click", closeKnowledgeModal);
document.getElementById("knowledgeCancel").addEventListener("click", closeKnowledgeModal);
knowledgeModal.addEventListener("click", (event) => {
  if (event.target === knowledgeModal) closeKnowledgeModal();
});

knowledgeForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = Object.fromEntries(new FormData(knowledgeForm).entries());
  knowledgeSubmit.disabled = true;
  knowledgeSubmit.textContent = "正在保存…";
  knowledgeMessage.textContent = "";
  knowledgeMessage.className = "form-message";

  try {
    const response = await fetch("/api/knowledge", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(data.detail) ? data.detail[0]?.msg : data.detail;
      throw new Error(detail || "保存失败");
    }

    knowledgeMessage.textContent = `保存成功，典故编号为 ${data.allusion_id}。`;
    knowledgeMessage.className = "form-message success";
    knowledgeForm.reset();
  } catch (error) {
    knowledgeMessage.textContent = `保存失败：${error.message}`;
    knowledgeMessage.className = "form-message error";
  } finally {
    knowledgeSubmit.disabled = false;
    knowledgeSubmit.textContent = "保存到知识库";
  }
});

document.getElementById("menuButton").addEventListener("click", () => {
  sidebar.classList.add("open");
  sidebarOverlay.classList.add("visible");
});
document.getElementById("sidebarClose").addEventListener("click", closeSidebar);
sidebarOverlay.addEventListener("click", closeSidebar);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !knowledgeModal.hidden) closeKnowledgeModal();
});

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const isDark = theme === "dark";
  themeIcon.textContent = isDark ? "☀" : "☾";
  themeLabel.textContent = isDark ? "浅色模式" : "深色模式";
  localStorage.setItem("duoshi-theme", theme);
}

themeButton.addEventListener("click", () => {
  applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
});

const savedTheme = localStorage.getItem("duoshi-theme");
const preferredTheme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
applyTheme(savedTheme || preferredTheme);

fetch("/api/health")
  .then((response) => response.json())
  .then((data) => { modelLabel.textContent = data.mode_label; })
  .catch(() => { modelLabel.textContent = "本地服务未连接"; });

resizeInput();
