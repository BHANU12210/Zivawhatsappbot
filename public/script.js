const send = document.getElementById("send");
const promptBox = document.getElementById("prompt");
const answerBox = document.getElementById("answer");
const lang = document.getElementById("lang");
const chips = document.querySelectorAll(".chip");

chips.forEach(c => c.onclick = () => {
  promptBox.value = c.dataset.q;
});

send.onclick = async () => {
  const text = promptBox.value.trim();
  if (!text) return;

  answerBox.innerHTML = "Thinking...";

  const res = await fetch("/chat", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({message: text, lang: lang.value})
  });

  const data = await res.json();
  answerBox.innerHTML = data.payload.answer || "No answer available.";
};
