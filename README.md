# Duoshi ✨

> An AI model for discovering allusions in Classical Chinese texts

---

## 🌟 What is Duoshi?

**Duoshi** is a lightweight and intuitive AI tool designed to help you:

* 🔍 Detect allusions in classical Chinese poetry and prose
* 🧠 Understand hidden meanings behind historical references
* 🔗 Explore intertextual connections across texts

Built on structured knowledge from traditional Chinese *leishu*類書 (encyclopedic compendia), Duoshi brings together classical scholarship and modern AI.

---

## 👨‍💻 Authors

Developed by:

* **Wang Zhitian** — Xiamen University Malaysia
* **Zhang Jie** — Southwest Jiaotong University

---

## ⚡ Why Duoshi?

Reading classical Chinese texts often requires deep familiarity with allusions.

Duoshi helps you:

* Save time on manual lookup
* Discover connections you might miss
* Support research, teaching, and learning

---

## 🧭 How it works (simple idea)

```
Text → Find allusions → Link meanings → Generate insights
```

That’s it. No complicated setup needed.

---

## 🚀 Use Cases

* 📖 Classical poetry analysis
* 🎓 Teaching Chinese literature
* 🧑‍🔬 Digital humanities research
* 🤖 AI-assisted annotation tools

---

## 📦 Installation

```bash
git clone https://github.com/ketsukun/Duoshi.git
cd Duoshi
pip install -r requirements.txt
```

---

## 💡 Quick Example

```python
from duoshi import DuoshiModel

model = DuoshiModel.load("duoshi-base")

text = "青松架短壑，赤脚踏层冰"
result = model.analyze(text)

print(result.allusions)
```

---

## 🛣️ Roadmap

* More *leishu* datasets
* Better semantic understanding
* Visualization tools
* Integration with more AI systems

---

## 📜 Citation

If you find this project useful, please cite:

> Wang Zhitian & Zhang Jie
> *Duoshi: An Allusion Recognition Model for Classical Chinese Texts*

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!

---

## 📄 License

MIT License
