# 🃏 Browser Card Game – Full Setup Guide

Welcome to the **Browser Card Game** project!  
This guide covers everything you need to **develop, test, and deploy** the application, from a zero setup to a working system — both locally and remotely.

---

## 📘 Overview

The project is built with:
- **Frontend:** React + Vite + Tailwind CSS  
- **Backend:** FastAPI (Python)  
- **Database/Cache:** Redis  
- **Containerization:** Docker & VS Code Dev Containers  

You can run everything **locally** using **VS Code Dev Containers**, or **deploy it to a remote server** (RHEL 9 / Ubuntu).

---

## 🧑‍💻 Part 1: Local Development Setup (VS Code Dev Container)

This is the easiest way to get started — all dependencies are handled for you inside a containerized environment.

### 🧩 Prerequisites

Install these three tools **on your local (Windows/macOS/Linux) machine**:

1. **[Docker Desktop](https://www.docker.com/products/docker-desktop/)**  
   > Make sure it’s running.
2. **[Visual Studio Code](https://code.visualstudio.com/)**
3. **VS Code Dev Containers Extension**  
   - Open VS Code  
   - Go to the **Extensions Marketplace** (`Ctrl+Shift+X`)  
   - Search for **“Dev Containers”** and install it

> 💡 You do **not** need to install Python, Node.js, or Redis locally — everything runs inside containers.

---

### 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
    ```

2. **Open in VS Code**

   ```bash
   code .
   ```

3. **Reopen in Dev Container**

   VS Code will detect the `.devcontainer` folder and show:

   > *“Folder contains a Dev Container configuration file. Reopen in container?”*

   Click **Reopen in Container**.

   * The first build will take a few minutes (Docker is creating the environment).
   * Subsequent launches are much faster.

4. **Start the services**

   ```bash
   docker compose up --build
   ```

5. **Access the app**

   * Frontend (React): [http://localhost:5173](http://localhost:5173)
   * Backend (FastAPI): [http://localhost:8000](http://localhost:8000)
   * API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

> ✅ Hot-reloading is enabled — changes in code instantly reflect in your running app.

---

### 🐞 One-Click Debugging

1. Open any backend file, e.g. `backend/app/main.py`
2. Set a breakpoint
3. Go to **Run and Debug** (`Ctrl+Shift+D`)
4. Select **Python: FastAPI** and click ▶️ (green play)

The debugger will attach to the running FastAPI server **inside the container**.

---

## 🏗️ Part 2: Project Structure & Initial Setup

If you’re creating a **new clone** or want to understand how the system is structured, this section shows how the backend and frontend fit together.

```
browser-card-game/
│
├── backend/            ← FastAPI backend
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   └── models/
│   └── requirements.txt
│
├── frontend/           ← React + Vite + Tailwind frontend
│   ├── src/
│   └── package.json
│
├── docker-compose.yml  ← Defines all services
├── .devcontainer/      ← VS Code container config
└── .vscode/            ← Debugger config
```

---

### 🐍 Backend Setup (FastAPI)

If your `backend` folder is missing, create it manually:

```bash
mkdir -p backend/app
touch backend/app/main.py
touch backend/requirements.txt
```

**`backend/requirements.txt`:**

```
fastapi
uvicorn[standard]
redis
debugpy
```

**`backend/app/main.py`:**

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "Backend is running!"}
```

To run the backend inside the container:

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

### ⚛️ Frontend Setup (Vite + React + Tailwind)

From the **project root**, create a frontend React app using Vite:

```bash
npm create vite@latest frontend -- --template react
```

Then install Tailwind CSS:

```bash
cd frontend
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
cd ..
```


Perfect ✅ — here’s the short, clear **compatibility note** you can insert right under the *Frontend Setup (Vite + React + Tailwind)* heading in your README.

I’ll show it in context so you can paste it directly:

---

> ⚠️ **Compatibility Note (Tailwind 4.x vs 3.x)**
> The latest **Tailwind CSS 4.x** releases introduce a new compiler and configuration system that are **not yet fully compatible** with the current **Vite + PostCSS + React** toolchain.
> If you install Tailwind 4.x (e.g., `"tailwindcss": "^4.1.16"`), you may see errors like:
>
> ```
> npm error could not determine executable to run
> ```
>
> This happens because Tailwind 4.x no longer exposes the same CLI entry point (`npx tailwindcss init -p` will fail).
>
> ✅ **Recommended fix:**
> Stick to the latest **Tailwind 3.x** series for now, which works flawlessly with Vite and PostCSS:
>
> ```json
> "devDependencies": {
>   "tailwindcss": "3.4.18",
>   "postcss": "8.4.47",
>   "autoprefixer": "10.4.20"
> }
> ```
>
> You can safely update to Tailwind 4.x once the ecosystem (Vite plugin and documentation) officially supports it.

#### Update `frontend/tailwind.config.js`

```js
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: { extend: {} },
  plugins: [],
}
```

#### Update `frontend/vite.config.js`

```js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    watch: {
      usePolling: true, // Enables hot reload inside Docker
    },
  },
})
```

#### Replace `frontend/src/index.css`

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

Then build dependencies:

```bash
cd frontend
npm install
cd ..
```

---

### 🧠 Understanding Container–Project Interaction

* **Your project files (code)** are stored **on your host machine**.
* **The container** mounts your project directory as a shared volume:

  * You edit files in VS Code → they instantly update in the container.
  * The container provides the environment (Python, Node.js, Redis, etc.).
* **Hot reload** works both for the backend and frontend, so you can develop seamlessly.

---

## 🌐 Part 3: Deploying to a Remote Server (RHEL 9 / Ubuntu)

For production or staging, deploy using Docker on a **headless Linux server**.

### 🧩 Prerequisites

* A fresh RHEL 9 or Ubuntu 22.04+ server
* SSH access and a user with `sudo` privileges
* Git installed:

  ```bash
  sudo dnf install git     # RHEL
  sudo apt install git     # Ubuntu
  ```

---

### 🐳 Install Docker & Docker Compose

#### RHEL 9

```bash
sudo dnf update -y
sudo dnf -y install dnf-utils
sudo dnf config-manager --add-repo https://download.docker.com/linux/rhel/docker-ce.repo
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
```

> Log out and back in to apply group changes.

#### Ubuntu 22.04+

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker $USER
```

---

### 🚀 Deploy the App

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
docker compose up -d --build
```

Now open:

* Frontend → `http://<server_ip>:5173`
* Backend → `http://<server_ip>:8000`

---

### Access from local browser

Now you will use the SSH Tunnel, as described in the tutorial, to connect your local machine to the secure application running on the server.

- On your LOCAL machine (in a new terminal):

Run the following ssh command. This command forwards your local ports 5173 and 8000 through an encrypted tunnel to the server's localhost:5173 and localhost:8000.

Replace username@your_server_ip with your server's credentials.

```bash
ssh -L 5173:localhost:5173 -L 8000:localhost:8000 username@your_server_ip
```

- Leave this terminal running. It is maintaining your secure tunnel.

---

### 🔥 Configure the Firewall {DO NOT DO EXCEPT FOR PRODUCITON !!!}

#### RHEL 9

```bash
sudo firewall-cmd --permanent --add-port=5173/tcp
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --reload
```

#### Ubuntu

```bash
sudo ufw allow ssh
sudo ufw allow 5173/tcp
sudo ufw allow 8000/tcp
sudo ufw enable
```

---

### ⚙️ Managing the Application

| Task                | Command                        |
| ------------------- | ------------------------------ |
| View logs           | `docker compose logs -f`       |
| Stop containers     | `docker compose down`          |
| Pull latest version | `git pull`                     |
| Rebuild & restart   | `docker compose up -d --build` |

---

## ✅ Summary

You now have:

* A **portable development environment** with VS Code Dev Containers
* A working **backend (FastAPI)** and **frontend (React + Tailwind)** setup
* A **ready-to-deploy** Dockerized system for production

Happy hacking! 🎮