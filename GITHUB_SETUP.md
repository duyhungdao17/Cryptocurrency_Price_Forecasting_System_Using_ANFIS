# GitHub Push Guide

## Chuẩn bị (Setup)

### 1. Repository đã tạo
Repository của bạn đã được tạo tại:
```
https://github.com/duyhungdao17/Cryptocurrency_Price_Forecasting_System_Using_ANFIS.git
```

---

## Cách sử dụng Script

### Option 1: PowerShell (Recommended for Windows)

```powershell
# Mở PowerShell ở thư mục project
cd "d:\SGU-Semester\NAMIV-HK2\CI\Crypto_Forecasting_Model"

# Chạy script với GitHub URL của bạn
.\push-to-github.ps1 "https://github.com/duyhungdao17/Cryptocurrency_Price_Forecasting_System_Using_ANFIS.git"
```

**Lưu ý:** Nếu gặp lỗi `cannot be loaded because running scripts is disabled`:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Rồi chạy lại script.

### Option 2: Command Prompt (CMD)

```cmd
REM Mở cmd ở thư mục project
cd d:\SGU-Semester\NAMIV-HK2\CI\Crypto_Forecasting_Model

REM Chạy batch script
push-to-github.bat https://github.com/duyhungdao17/Cryptocurrency_Price_Forecasting_System_Using_ANFIS.git
```

### Option 3: Manual Git Commands

```bash
cd d:\SGU-Semester\NAMIV-HK2\CI\Crypto_Forecasting_Model

git init
git add .
git commit -m "Initial commit: Crypto Forecasting Model with ANFIS, LSTM, and ANN"
git branch -M main
git remote add origin https://github.com/duyhungdao17/Cryptocurrency_Price_Forecasting_System_Using_ANFIS.git
git push -u origin main
```

---

## Cấu hình GitHub Credentials

### Nếu dùng HTTPS (cần Personal Access Token)

1. **Tạo Personal Access Token:**
   - GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Click **Generate new token (classic)**
   - Scope: Chọn `repo`
   - Copy token

2. **Khi git yêu cầu password:**
   - Username: `your-github-username`
   - Password: Dán **Personal Access Token** vừa tạo (không phải password GitHub)

### Nếu dùng SSH (Recommended)

1. **Tạo SSH Key:**
```bash
ssh-keygen -t ed25519 -C "your-email@example.com"
```
Nhấn Enter để dùng default locations.

2. **Thêm vào SSH agent:**
```bash
ssh-agent
ssh-add $env:USERPROFILE\.ssh\id_ed25519
```

3. **Thêm Public Key vào GitHub:**
   - Copy file `$env:USERPROFILE\.ssh\id_ed25519.pub`
   - GitHub → Settings → SSH and GPG keys → New SSH key
   - Paste vào

4. **Sử dụng SSH URL:**
```
git@github.com:duyhungdao17/Cryptocurrency_Price_Forecasting_System_Using_ANFIS.git
```

---

## .gitignore - Những file bị bỏ qua

Script tự động tạo `.gitignore` với:
- `setup_env.bat` ✓
- `setup_env.ps1` ✓
- `__pycache__/`
- `*.pyc`
- `checkpoints/` (weights)
- `Plot/` (generated images)
- `data/` (raw data)
- `Summary/` (output files)
- `.env` (credentials)
- Và nhiều file khác không cần trên GitHub

---

## Kiểm tra kết quả

Sau khi push thành công, truy cập:
```
https://github.com/duyhungdao17/Cryptocurrency_Price_Forecasting_System_Using_ANFIS
```

Bạn sẽ thấy:
- ✓ `main.py`, `README.md`, requirements.txt
- ✓ Thư mục `Model/`, `Crawling/`, `UI/`
- ✓ **Không** có: setup files, data files, checkpoints, plots

---

## Lệnh hữu ích sau này

```bash
# Cập nhật changes mới
git add .
git commit -m "Your message"
git push

# Xem status
git status

# Xem log commits
git log --oneline

# Tạo branch mới
git checkout -b feature/your-feature
git push -u origin feature/your-feature
```

---

## Troubleshooting

**❌ "fatal: destination path 'xxx' already exists"**
- Repository đã tồn tại. Xóa `.git` folder: `Remove-Item .git -Force -Recurse`

**❌ "Permission denied (publickey)"**
- SSH key không đúng. Kiểm tra: `ssh -T git@github.com`

**❌ "Support for password authentication was removed"**
- Dùng Personal Access Token thay vì password

**❌ "You may want to first integrate the remote changes"**
- Remote repo có file khác. Chạy: `git pull --rebase origin main`

---

## ⚡ Quick Start (Dành cho bạn)

Bạn đã tạo repository tại: `https://github.com/duyhungdao17/Cryptocurrency_Price_Forecasting_System_Using_ANFIS.git`

**Chạy ngay bây giờ:**

```powershell
cd "d:\SGU-Semester\NAMIV-HK2\CI\Crypto_Forecasting_Model"
.\push-to-github.ps1 "https://github.com/duyhungdao17/Cryptocurrency_Price_Forecasting_System_Using_ANFIS.git"
```

Hoặc dùng CMD:
```cmd
cd d:\SGU-Semester\NAMIV-HK2\CI\Crypto_Forecasting_Model
push-to-github.bat https://github.com/duyhungdao17/Cryptocurrency_Price_Forecasting_System_Using_ANFIS.git
```

Xong! Project sẽ được đẩy lên GitHub 🚀
