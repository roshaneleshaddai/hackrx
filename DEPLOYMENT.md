# Deployment Guide

## 🚀 Platform Options

### 1. **Railway** (Recommended)

**Best for Python/Flask + AI apps**

#### Setup:

1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your `hackrx` repository
5. Add environment variable: `MISTRAL_API_KEY=your_key_here`
6. Deploy!

**Advantages:**

- ✅ No size limits
- ✅ Great Python support
- ✅ Free tier available
- ✅ Automatic deployments
- ✅ Custom domains

---

### 2. **Render**

**Good alternative with free tier**

#### Setup:

1. Go to [render.com](https://render.com)
2. Sign up with GitHub
3. Click "New" → "Web Service"
4. Connect your GitHub repo
5. Configure:
   - **Name**: `hackrx`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
6. Add environment variable: `MISTRAL_API_KEY`
7. Deploy!

**Advantages:**

- ✅ Free tier
- ✅ Easy setup
- ✅ Good Python support

---

### 3. **Heroku**

**Classic choice for Flask apps**

#### Setup:

1. Install Heroku CLI
2. Run commands:

```bash
heroku create your-app-name
git push heroku main
heroku config:set MISTRAL_API_KEY=your_key_here
```

**Advantages:**

- ✅ Mature platform
- ✅ Excellent documentation
- ✅ Good free tier (with sleep)

---

### 4. **DigitalOcean App Platform**

**Reliable and scalable**

#### Setup:

1. Go to [DigitalOcean](https://cloud.digitalocean.com/apps)
2. Create new app from GitHub
3. Select Python environment
4. Add environment variables
5. Deploy!

**Advantages:**

- ✅ Reliable infrastructure
- ✅ Good pricing
- ✅ No cold starts

---

### 5. **Google Cloud Run**

**Serverless with no cold starts**

#### Setup:

1. Install Google Cloud CLI
2. Run commands:

```bash
gcloud run deploy hackrx --source .
gcloud run services update hackrx --set-env-vars MISTRAL_API_KEY=your_key
```

**Advantages:**

- ✅ Pay per use
- ✅ No cold starts
- ✅ Great for AI apps

---

## 🔧 Environment Variables

All platforms need this environment variable:

```
MISTRAL_API_KEY=your_mistral_api_key_here
```

## 📝 API Usage

Once deployed, your API will be available at:

```
https://your-app-name.railway.app/hackrx/run
```

### Example Request:

```bash
curl -X POST https://your-app-name.railway.app/hackrx/run \
  -H "Authorization: Bearer 3ca0894d22ac6bf6daf7d8323b1e77d69241f8b2810b9bee667a0a14969ffb48" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": ["What is the main topic?"]
  }'
```

## 🎯 Recommendation

**Use Railway** - it's the best choice for your Flask + AI app because:

- No size limitations
- Excellent Python support
- Free tier available
- Easy deployment from GitHub
- Perfect for AI/ML applications
