# 🧮 OpenAI Token Counter & Cost Calculator

A comprehensive **Streamlit web application** for calculating token counts and costs across **all OpenAI model types** including text models, audio processing, image generation, embeddings, fine-tuning, and specialized tools. This application provides real-time cost analysis with support for batch API, cached inputs, and detailed cost breakdowns.

## 🆕 **What's New in Version 2.0**

### **📁 File-Based Model Management**
- **Zero Hardcoded Models**: All model data is loaded from JSON files
- **Dynamic Configuration**: Centralized config system for easy management
- **Real-time Updates**: Add/edit/delete models without code changes
- **Automatic Backup**: Built-in backup and restore functionality
- **Data Validation**: Comprehensive validation with error reporting

### **⚙️ Model Management Interface**
- **Complete CRUD Operations**: Add, view, edit, and delete models through the UI
- **Category Management**: Organize models by categories and pricing types
- **Import/Export**: Bulk operations for model data
- **Backup/Restore**: Automated backups with easy restore functionality
- **Validation Tools**: Built-in validation with detailed error reporting

## 🚀 Features Overview

### 📊 **Complete Model Coverage**
- **Text Models**: GPT-4o, GPT-3.5 Turbo, o1-preview, o1-mini, ChatGPT-4o
- **Audio Models**: GPT-4o Audio, Realtime Audio processing
- **Speech Models**: Whisper (transcription), TTS (text-to-speech)
- **Embeddings**: text-embedding-3-small/large, ada-002
- **Image Generation**: DALL-E 3, DALL-E 2
- **Fine-tuning**: Custom model training and inference
- **Special Tools**: Web search, code interpreter, file search
- **Moderation**: Free content moderation models

### 💰 **Advanced Cost Analysis**
- **Real-time pricing** based on official OpenAI rates
- **Batch API discounts** (50% savings)
- **Cached input discounts** (50% savings)
- **Detailed cost breakdowns** for every model type
- **Multi-format file processing** with automatic token counting

### 🔧 **File Processing Support**
- **Text files**: .txt, .md, .csv
- **Documents**: .docx, .pdf, .pptx
- **Spreadsheets**: .xlsx, .xls
- **Images**: .png, .jpg, .jpeg (for context)

## 📋 Requirements

### System Requirements
- **Python 3.8+**
- **4GB RAM minimum**
- **Web browser** (Chrome, Firefox, Safari, Edge)

### Dependencies
All dependencies are automatically installed via `requirements.txt`:
```
streamlit>=1.36.0      # Web framework
tiktoken>=0.7.0        # Token counting
pandas>=2.0.0          # Data processing
numpy>=1.24.0          # Numerical operations
python-docx>=1.1.0     # Word documents
PyPDF2>=3.0.0          # PDF processing
pdfplumber>=0.10.0     # Advanced PDF extraction
openpyxl>=3.1.0        # Excel files
python-pptx>=0.6.21    # PowerPoint files
Pillow>=10.0.0         # Image processing
plotly>=5.17.0         # Interactive charts
chardet>=5.2.0         # Text encoding detection
```

## 🚀 Quick Start Guide

### Option 1: Direct Launch (Recommended)
```bash
# 1. Clone or download the project
git clone <repository-url>
cd tokken_counter

# 2. Run the application (auto-installs dependencies)
python run.py
```

### Option 2: Manual Setup
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Launch the application
streamlit run app.py
```

### Option 3: Test Before Running
```bash
# Run comprehensive tests first
python test_app.py

# Then launch the application
python run.py
```

**The application will automatically open in your web browser at `http://localhost:8501`**

## 📁 **Model Management System**

### **🎯 Overview**
Version 2.0 introduces a comprehensive file-based model management system that eliminates hardcoded model data and provides full control over model configurations through JSON files.

### **📂 File Structure**
```
tokken_counter/
├── pricing_data/
│   ├── config.json                    # Central configuration
│   ├── text_models.json              # Text-based models
│   ├── audio_models.json             # Audio processing models
│   ├── transcription_models.json     # Speech-to-text models
│   ├── embeddings_models.json        # Embedding models
│   ├── fine_tuning_models.json       # Fine-tuning models
│   ├── moderation_models.json        # Content moderation models
│   ├── image_generation_models.json  # Image generation models
│   ├── web_search_models.json        # Web search models
│   └── built_in_tools.json          # Built-in tools
├── logs/                             # Application logs
├── backups/                          # Automatic backups
├── model_manager.py                  # Core model management
├── model_utils.py                    # Utility functions
└── model_management_ui.py           # Streamlit UI components
```

### **⚙️ Model Management Interface**

Access the model management interface through the **"Model Management"** tab in the application.

#### **📊 View Models**
- Browse all models by category
- View detailed model information
- Search and filter capabilities
- Export model data as CSV/JSON

#### **➕ Add New Models**
1. Select existing category or create new one
2. Choose pricing type (standard, audio, embeddings, etc.)
3. Fill in model details (costs, context window, etc.)
4. Automatic validation before saving
5. Real-time availability in the application

#### **✏️ Edit Existing Models**
1. Select category and model
2. Modify any field (name, costs, features)
3. Validation ensures data integrity
4. Automatic backup before changes
5. Immediate updates across the application

#### **🗑️ Delete Models**
1. Select model to remove
2. Preview model details
3. Confirmation required
4. Automatic backup created
5. Safe deletion with rollback option

#### **📤 Import/Export Operations**
- **Export**: Download model data as JSON
- **Import**: Upload and merge model data
- **Batch Operations**: Manage multiple models
- **Format Validation**: Ensures data integrity

#### **💾 Backup & Restore**
- **Automatic Backups**: Created before any changes
- **Manual Backups**: On-demand backup creation
- **Restore Points**: Roll back to any previous state
- **Backup Management**: Automatic cleanup of old backups

### **🔧 Configuration Management**

#### **config.json Structure**
```json
{
  "application": {
    "name": "OpenAI Token Counter & Cost Calculator",
    "version": "2.0.0"
  },
  "data_sources": {
    "models_directory": "pricing_data",
    "model_files": [...]
  },
  "validation_rules": {
    "required_fields": {...},
    "field_types": {...},
    "cost_ranges": {...}
  },
  "ui_settings": {
    "enable_model_management": true,
    "items_per_page": 20
  },
  "backup_settings": {
    "auto_backup": true,
    "max_backups": 30
  }
}
```

### **📝 Adding New Models**

#### **Method 1: Using the UI (Recommended)**
1. Go to **Model Management** tab
2. Select **"Add Model"**
3. Fill in the form fields
4. Click **"Add Model"**

#### **Method 2: Direct File Editing**
1. Open the appropriate JSON file in `pricing_data/`
2. Add your model following the existing structure
3. Restart the application to load changes

#### **Example Model Structure**
```json
{
  "your-model-id": {
    "name": "Your Model Name",
    "description": "Model description",
    "input_cost": 2.50,
    "output_cost": 10.00,
    "cached_input_cost": 1.25,
    "context_window": 128000,
    "category": "Your Category",
    "pricing_type": "standard",
    "has_cached": true,
    "has_batch": true
  }
}
```

### **🔍 Model Validation**

The system includes comprehensive validation:

#### **Required Fields by Pricing Type**
- **Standard Models**: name, description, input_cost, output_cost, context_window, category
- **Audio Models**: text_input_cost, text_output_cost, audio_input_cost, audio_output_cost
- **Embeddings**: input_cost, dimensions
- **Per-minute Models**: cost_per_minute
- **Free Models**: is_free flag

#### **Data Type Validation**
- String fields: name, description, category
- Numeric fields: all cost values, context_window, dimensions
- Boolean fields: has_cached, has_batch, is_free

#### **Range Validation**
- Cost values: $0.00 - $1000.00 per 1M tokens
- Context windows: 1,000 - 2,000,000 tokens

### **🚨 Error Handling & Recovery**

#### **Automatic Recovery**
- **Fallback Models**: Emergency models if files are corrupted
- **Backup Restoration**: Automatic rollback on critical errors
- **Validation Warnings**: Non-blocking warnings for minor issues

#### **Manual Recovery**
1. **Check Logs**: Review `logs/model_manager.log`
2. **Restore Backup**: Use the restore interface
3. **Validate Data**: Run validation tools
4. **Emergency Reset**: Restore from emergency fallback

## 📖 Complete Usage Guide

### 🗂️ **Tab 1: Text Input**
**Purpose**: Calculate costs for text-based models with manual input

#### Text & Chat Models
1. **Select Model Category** (Flagship, Small Models, Reasoning, etc.)
2. **Choose Specific Model** (GPT-4o, o1-mini, etc.)
3. **Enter your text** in the text area
4. **Set output tokens** estimate
5. **Toggle options**:
   - ✅ Use Cached Input (50% discount)
   - ✅ Use Batch API (50% discount)

**Results Display**:
- 📊 Input/Output token counts
- 💰 Total cost breakdown
- ⚠️ Context window warnings

#### Audio Processing Models
1. **Select Audio Model** (GPT-4o Audio, Realtime)
2. **Enter text component**
3. **Specify audio tokens**:
   - Audio Input Tokens
   - Audio Output Tokens
4. **Enable caching** if supported

#### Embeddings Models
1. **Select Embeddings Model** (text-embedding-3-small/large)
2. **Enter text for embedding**
3. **Choose Batch API** for discounts
4. **View dimensions** and cost per 1K tokens

#### Speech Models
**Text-to-Speech (TTS)**:
- Enter text to convert
- Automatic character counting
- Real-time cost calculation

**Speech-to-Text (Whisper)**:
- Input audio duration in minutes
- Per-minute pricing display

#### Fine-tuning Models
1. **Select fine-tuning model**
2. **Training Phase**:
   - Training tokens OR training hours
3. **Inference Phase**:
   - Input/output token estimates
4. **Options**:
   - Cached input support
   - Data sharing discounts

### 📁 **Tab 2: File Upload**
**Purpose**: Process files and calculate costs automatically

#### Supported File Types
| File Type | Extensions | Processing Method |
|-----------|------------|-------------------|
| Text Files | .txt, .md, .csv | Direct text extraction |
| Word Documents | .docx | python-docx library |
| PDF Files | .pdf | PyPDF2 + pdfplumber |
| Excel Files | .xlsx, .xls | openpyxl + xlrd |
| PowerPoint | .pptx | python-pptx |

#### Upload Process
1. **Drag & drop** or **browse** for files
2. **Select model** for cost calculation
3. **Choose pricing options** (cached, batch)
4. **View results**:
   - Extracted text preview
   - Token count
   - Cost estimation
   - Processing time

### 📊 **Tab 3: Analytics**
**Purpose**: Compare models and analyze cost efficiency

#### Model Comparison
1. **Enter comparison text**
2. **Set output token estimate**
3. **View interactive chart** comparing:
   - GPT-4o Mini (most cost-effective)
   - GPT-4o (balanced performance)
   - o1-Mini (for reasoning tasks)

#### Analytics Features
- 📈 **Cost comparison charts**
- 📋 **Model performance tables**
- 💡 **Optimization recommendations**

### 🗂️ **Tab 7: Model Management**
**Purpose**: Comprehensive model data management interface

#### Model Operations
1. **View Models**: Browse and search all models by category
2. **Add Models**: Create new models with guided forms
3. **Edit Models**: Modify existing model properties
4. **Delete Models**: Remove models with confirmation and backup
5. **Import/Export**: Bulk operations for model data
6. **Backup/Restore**: Manage backup files and restoration
7. **Validate Data**: Check data integrity and fix issues

#### Management Features
- **Real-time Updates**: Changes apply immediately
- **Automatic Backups**: Created before any modifications
- **Data Validation**: Comprehensive error checking
- **Category Management**: Organize models by type
- **Bulk Operations**: Handle multiple models efficiently

### 🔧 **Tab 4: Tools & Special Models**
**Purpose**: Calculate costs for specialized OpenAI services

#### Image Generation (DALL-E)
**DALL-E 3**:
- **Resolutions**: 1024×1024, 1024×1792, 1792×1024
- **Quality**: Standard ($0.040), HD ($0.080-$0.120)
- **Pricing**: Per-image basis

**DALL-E 2**:
- **Resolutions**: 256×256 ($0.016), 512×512 ($0.018), 1024×1024 ($0.020)

#### Web Search Models
1. **Select search model**
2. **Enter number of calls**
3. **Choose context size**:
   - Small, Medium, Large
4. **Per-1K-calls pricing**

#### Built-in Tools
**Code Interpreter**:
- Per-container pricing
- Session-based costs

**File Search Storage**:
- GB-day pricing model
- Free tier included

**File Search Calls**:
- Per-1K-calls pricing

### 🔍 **Tab 5: System Status**
**Purpose**: Monitor application health and validation

#### System Monitoring
- 📄 **Pricing files status** (9 files loaded)
- 📊 **Model counts by category**
- 💰 **Pricing types distribution**
- ⚠️ **Issues and warnings**

#### Interactive Charts
- Bar chart of model distribution
- Pie chart of pricing types
- Real-time validation results

### 🧪 **Tab 6: Model Testing Hub**
**Purpose**: Comprehensive testing for all model types

#### Testing Categories

##### 🤖 Text & Chat Models
- **Full token counting**
- **Cost optimization options**
- **Context window validation**
- **Real-time cost updates**

##### 🎵 Audio & Speech Models
**Three Subcategories**:

1. **🎙️ Audio Processing**
   - Text + Audio token inputs
   - Cached input support
   - Real-time cost calculation

2. **🗣️ Text-to-Speech**
   - Character-based pricing
   - Live character counting
   - Model comparison (TTS vs TTS-HD)

3. **📝 Speech-to-Text**
   - Per-minute pricing
   - Audio duration input
   - Whisper model testing

##### 🔤 Embeddings Models
- **Text input processing**
- **Dimension information**
- **Batch API discounts**
- **Cost per 1K tokens**

##### 🎨 Image Generation
- **Resolution selection**
- **Quality options**
- **Batch image generation**
- **Sample prompt suggestions**

##### 🔍 Search & Tools
**Web Search**:
- Call volume testing
- Context size impact
- Cost per 1K calls

**Built-in Tools**:
- Code interpreter sessions
- File search storage (GB-days)
- File search calls

##### 🎓 Fine-tuning Models
- **Training cost calculation**
- **Inference cost estimation**
- **Data sharing options**
- **Cached input support**

##### 🆓 Free Models
- **Moderation model testing**
- **Token counting for free services**

## 💰 Cost Calculation Guide

### Standard Pricing Formula
```
Input Cost = (Input Tokens ÷ 1,000,000) × Input Rate
Output Cost = (Output Tokens ÷ 1,000,000) × Output Rate
Total Cost = Input Cost + Output Cost
```

### Discount Applications
```
Cached Input = Standard Input Cost × 0.5
Batch API = Standard Cost × 0.5
Combined Discount = Standard Cost × 0.25 (maximum savings)
```

### Model-Specific Calculations

#### Text Models
```python
# GPT-4o Example
input_cost = (tokens / 1_000_000) * 2.50
output_cost = (tokens / 1_000_000) * 10.00
```

#### Audio Models
```python
# GPT-4o Audio
text_cost = (text_tokens / 1_000_000) * 2.50
audio_cost = (audio_tokens / 1_000_000) * 100.00
```

#### Embeddings
```python
# text-embedding-3-small
cost = (input_tokens / 1_000_000) * 0.02
```

#### TTS/Whisper
```python
# TTS-1
cost = (characters / 1_000_000) * 15.00

# Whisper-1
cost = minutes * 0.006
```

#### Image Generation
```python
# DALL-E 3 (1024x1024, standard)
cost = num_images * 0.040
```

#### Fine-tuning
```python
# Training
training_cost = (training_tokens / 1_000_000) * 25.00

# Inference (higher rates)
inference_cost = (tokens / 1_000_000) * 3.75
```

## 🏗️ Architecture Overview

### File Structure
```
tokken_counter/
├── app.py                 # Main Streamlit application (1,988 lines)
├── requirements.txt       # Python dependencies
├── run.py                # Application launcher
├── test_app.py           # Comprehensive test suite
├── README.md             # This documentation
├── .streamlit/
│   └── config.toml       # Streamlit configuration
└── pricing_data/         # JSON pricing files
    ├── text_models.json          # GPT models
    ├── audio_models.json         # Audio processing
    ├── transcription_models.json # Speech models
    ├── embeddings_models.json    # Embedding models
    ├── fine_tuning_models.json   # Fine-tuning
    ├── image_generation_models.json  # DALL-E
    ├── web_search_models.json    # Search models
    ├── built_in_tools.json       # Special tools
    └── moderation_models.json    # Free models
```

### Core Classes

#### `OpenAIModels`
**Purpose**: Central model management and cost calculation
```python
class OpenAIModels:
    def __init__(self)                    # Initialize and load all pricing data
    def load_pricing_data(self)           # Load from JSON files
    def calculate_cost_by_category(self)  # Route to appropriate calculator
    def calculate_text_model_cost(self)   # Standard text models
    def calculate_audio_model_cost(self)  # Audio processing
    def calculate_embeddings_cost(self)   # Embeddings
    # ... 12 specialized cost calculators
```

#### `TokenCalculator`
**Purpose**: Token counting and cost calculation utilities
```python
class TokenCalculator:
    @staticmethod
    def count_tokens(text, model)         # tiktoken-based counting
    @staticmethod  
    def calculate_cost(tokens, model_info) # Legacy cost calculation
```

#### `FileProcessor`  
**Purpose**: Multi-format file processing
```python
class FileProcessor:
    @staticmethod
    def detect_encoding(file_bytes)       # Smart encoding detection
    @staticmethod
    def extract_text_from_docx(file)      # Word documents
    @staticmethod
    def extract_text_from_pdf(file)       # PDF processing
    # ... processors for Excel, PowerPoint, etc.
```

### Pricing Data Structure

Each JSON file follows a consistent structure:
```json
{
  "model_category": {
    "model_id": {
      "name": "Display Name",
      "description": "Model description",
      "input_cost": 2.50,           // Per 1M tokens
      "output_cost": 10.00,         // Per 1M tokens
      "context_window": 128000,     // Token limit
      "category": "Model Category",
      "pricing_type": "standard",   // Calculation method
      "has_cached": true,           // Supports caching
      "has_batch": true             // Supports batch API
    }
  }
}
```

## 🎯 Optimization Tips

### Cost Optimization Strategies

#### 1. Model Selection
- **Simple tasks**: Use GPT-4o Mini or GPT-3.5 Turbo
- **Complex reasoning**: Use o1-preview or o1-mini
- **Balanced performance**: Use GPT-4o
- **Creative tasks**: Use GPT-4.5 preview (when available)

#### 2. Batch API Usage
```python
# Use for non-real-time workloads
use_batch = True  # 50% cost reduction
```

#### 3. Cached Inputs
```python
# For repeated content processing
use_cached = True  # 50% cost reduction on input
```

#### 4. Context Window Management
- **Monitor utilization**: Keep under 80% for optimal performance
- **Break large content**: Split into smaller chunks
- **Use appropriate models**: Don't use large context models for small tasks

#### 5. Output Token Control
- **Optimize prompts**: Request concise responses
- **Set max_tokens**: Limit output length
- **Use specific instructions**: Reduce unnecessary verbosity

### Performance Optimization

#### File Processing
```python
# Optimal file sizes
Text files: < 10MB
PDF files: < 50MB  
Excel files: < 25MB
```

#### Memory Management
- Process large files in chunks
- Clear session state regularly
- Use appropriate data types

## 🔧 Configuration

### Streamlit Configuration (`.streamlit/config.toml`)
```toml
[server]
port = 8501
address = "localhost"
maxUploadSize = 200         # MB
enableCORS = false

[theme]
primaryColor = "#1E90FF"    # Blue accent
backgroundColor = "#FFFFFF"  # White background
font = "sans serif"

[browser]
gatherUsageStats = false    # Privacy focused
```

### Environment Variables
```bash
# Optional: Set custom port
export STREAMLIT_SERVER_PORT=8501

# Optional: Disable telemetry
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

## 🧪 Testing

### Automated Testing
Run comprehensive tests before deployment:
```bash
python test_app.py
```

**Test Coverage**:
- ✅ Import validation (12 modules)
- ✅ tiktoken functionality  
- ✅ File processing capabilities
- ✅ Model data integrity
- ✅ Token calculation accuracy
- ✅ Streamlit component functionality

### Manual Testing Checklist
- [ ] Text input processing
- [ ] File upload functionality
- [ ] Cost calculations for each model type
- [ ] Pricing option toggles
- [ ] Chart generation
- [ ] Error handling

## 🚨 Troubleshooting

### Model Management Issues

#### Model Manager Not Loading
```bash
# Check required files exist
ls pricing_data/config.json
ls pricing_data/*.json

# Verify file permissions
chmod 644 pricing_data/*.json

# Check logs for detailed errors
tail -f logs/model_manager.log
```

#### Validation Failures
1. **Go to Model Management → Validate Data**
2. **Review error messages in detail**
3. **Use Edit Model to fix issues**
4. **Restore from backup if data is corrupted**

#### Import/Export Issues
```bash
# Ensure proper JSON format
python -c "import json; json.load(open('your_file.json'))"

# Check file permissions
chmod 644 your_export_file.json
```

#### Backup/Restore Problems
```bash
# Create backup directory
mkdir -p backups

# Check backup file integrity
python -c "import json; json.load(open('backups/backup_file.json'))"

# Verify restore permissions
chmod 755 backups/
```

### Common Issues

#### Model Data Loading Errors
```bash
# Solution 1: Recreate config file
# The app will auto-generate if missing
rm pricing_data/config.json

# Solution 2: Validate JSON files
python -c "
import json, os
for f in os.listdir('pricing_data'):
    if f.endswith('.json'):
        try:
            json.load(open(f'pricing_data/{f}'))
            print(f'✅ {f}')
        except: print(f'❌ {f}')
"
```

#### Import Errors
```bash
# Solution: Install all dependencies
pip install -r requirements.txt

# Verify specific model management dependencies
pip install --upgrade streamlit pandas
```

#### Port Already in Use
```bash
# Solution: Use different port
streamlit run app.py --server.port 8502
```

#### Memory Issues with Large Files
```bash
# Solution: Increase upload limit in config.toml
maxUploadSize = 500
```

#### Token Counting Errors
```bash
# Solution: Check tiktoken installation
pip install --upgrade tiktoken

# Verify encoding compatibility
python -c "import tiktoken; print(tiktoken.list_encoding_names())"
```

### Error Codes

| Error | Cause | Solution |
|-------|-------|----------|
| ModuleNotFoundError | Missing dependency | `pip install -r requirements.txt` |
| FileNotFoundError | Missing pricing data | Check `pricing_data/` folder existence |
| EncodingError | Invalid file encoding | Use UTF-8 encoded files |
| MemoryError | File too large | Reduce file size or increase RAM |
| ValidationError | Invalid model data | Use Model Management validation tools |
| PermissionError | File access denied | Check file/directory permissions |
| JSONDecodeError | Malformed JSON | Validate JSON syntax with online tools |
| ImportError | Module loading failed | Restart application or reinstall dependencies |

### Recovery Procedures

#### Complete Data Recovery
1. **Check emergency fallback**: App automatically loads minimal models
2. **Restore from backup**: Use Model Management → Backup/Restore
3. **Reset to defaults**: Delete pricing_data/ and restart (auto-regenerates)
4. **Manual restoration**: Copy from original installation

#### Fixing Corrupted Models
1. **Use validation tool**: Model Management → Validate Data
2. **Edit problematic models**: Fix validation errors through UI
3. **Remove invalid models**: Delete through Model Management
4. **Import clean data**: Use export from working installation

## 📈 Future Enhancements

### Planned Features
- 🔐 **API key integration** for real-time pricing
- 📊 **Usage analytics dashboard**  
- 💾 **Cost history tracking**
- 🔄 **Automated pricing updates**
- 🌐 **Multi-language support**
- 📱 **Mobile-responsive design**

### Extensibility
The application is designed for easy extension:
- Add new model types in `pricing_data/`
- Implement new cost calculators in `OpenAIModels`
- Create custom file processors in `FileProcessor`
- Add new UI tabs in the main application

## 📄 License & Credits

### License
This project is open-source and available under the MIT License.

### Credits
- **OpenAI**: For providing the models and pricing information
- **Streamlit**: For the excellent web framework
- **tiktoken**: For accurate token counting
- **Community contributors**: For testing and feedback

### Acknowledgments
- Pricing data accuracy verified against OpenAI's official documentation
- Regular updates to maintain pricing synchronization
- Community-driven feature requests and improvements

---

## 🆘 Support

### Getting Help
1. **Check this README** for comprehensive guidance
2. **Run test suite** to diagnose issues: `python test_app.py`
3. **Review error messages** in the Streamlit interface
4. **Check system requirements** and dependencies

### Contributing
We welcome contributions! Areas for improvement:
- Additional file format support
- Enhanced cost optimization algorithms
- New visualization features
- Performance optimizations
- Documentation improvements

---

**Made with ❤️ for the OpenAI community** | **Version 2.0** | **Last Updated: 2024** 