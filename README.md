# 🩺 AI-DRIVE: Diabetic Eye Detection System

A comprehensive AI-powered web application for detecting diabetic eye diseases using fundus and OCT image analysis.

## ✨ Features

- **🔍 Multi-Modal Analysis**: Supports both Fundus and OCT image analysis
- **🤖 AI-Powered Diagnosis**: Uses TensorFlow models for accurate disease detection
- **📊 Medical Conditions Detected**:
  - Cataract
  - Diabetic Retinopathy
  - Glaucoma
  - Diabetic Macular Edema (DME)
  - Normal eye conditions
- **📄 PDF Reports**: Generate professional medical reports
- **🧠 AI Medical Advice**: Google Gemini integration for medical recommendations
- **👤 User Management**: Authentication and patient record management
- **📱 Responsive Design**: Modern dark-themed UI with multi-language support
- **🐳 Docker Ready**: Containerized deployment with Docker
- **⚡ CI/CD**: GitHub Actions workflow for automated deployment

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Git LFS (for model files)
- Docker (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/pradeep1161/diabetic-eye-detection.git
   cd diabetic-eye-detection
   ```

2. **Set up environment**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env file with your configuration
   # Add your GEMINI_API_KEY and other settings
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   - Open your browser to `http://localhost:8501`
   - Navigate to `/diagnosis` for the main application

## 🐳 Docker Deployment

### Using Docker Compose
```bash
docker-compose up -d
```

### Using Docker directly
```bash
docker build -t ai-drive-diabetic-eye-detection .
docker run -p 8501:8501 ai-drive-diabetic-eye-detection
```

## 📁 Project Structure

```
diabetic-eye-detection/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
├── .github/workflows/    # CI/CD workflows
├── models/               # ML models (Git LFS)
│   ├── fundus_model.h5   # Fundus analysis model
│   └── oct_model.h5      # OCT analysis model
├── templates/            # HTML templates
├── static/               # CSS, JS, and static files
├── assets/               # Sample images
└── database/             # Database files
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file with the following variables:

```env
SECRET_KEY=your-secret-key
GEMINI_API_KEY=your-gemini-api-key
DATABASE_URL=sqlite:///patients.db
PORT=8501
FLASK_ENV=development
```

### Model Files
- Model files are stored using Git LFS due to their large size (>100MB each)
- Ensure Git LFS is installed: `git lfs install`
- Models are automatically downloaded when cloning the repository

## 🏥 Medical Disclaimer

**IMPORTANT**: This application is for research and educational purposes only. It is not intended to replace professional medical diagnosis, treatment, or advice. Always consult with qualified healthcare professionals for proper medical care.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow team for the ML framework
- Google Gemini for AI medical advice
- Flask community for the web framework
- All contributors and testers

## 📞 Support

For support, email [tallapallypradeep116@gmail.com] or create an issue in this repository.

---

**Made with ❤️ for better healthcare through AI**
