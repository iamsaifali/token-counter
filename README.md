# 🧮 OpenAI Token Counter & Cost Calculator

A comprehensive **Streamlit web application** for calculating token counts and costs across **all OpenAI model types** including text models, audio processing, image generation, embeddings, fine-tuning, and specialized tools. This application provides real-time cost analysis with support for batch API, cached inputs, and comprehensive model management.

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
- **Text files**: .txt, .md, .csv, .py, .js, .html, .css, .json, .xml
- **Documents**: .docx, .pdf, .pptx
- **Spreadsheets**: .xlsx, .xls

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
xlrd>=2.0.1            # Excel file reading
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

---

## 📖 **COMPLETE USER INTERFACE GUIDE**

This guide documents **every single UI element** in the application. No feature, button, dropdown, or option is omitted.

---

## 🎯 **MAIN APPLICATION HEADER**

### **Application Title**
- **Location**: Top of the page
- **Display**: "🧮 OpenAI Token Counter & Cost Calculator"
- **Purpose**: Identifies the application and its main function

### **Subtitle**
- **Text**: "Calculate token counts and costs for all OpenAI models with support for different pricing categories."
- **Purpose**: Brief description of application capabilities

---

## 📋 **SIDEBAR CONFIGURATION PANEL**

### **Configuration Header**
- **Label**: "⚙️ Configuration"
- **Purpose**: Groups all model selection and pricing options

### **Model Category Dropdown**
- **Label**: "Select Model Category"
- **Type**: Dropdown/Selectbox
- **Options**: Dynamically loaded categories including:
  - Audio Models
  - Built-in Tools
  - Embedding Models  
  - Fine-tuning Models
  - Flagship Models
  - Image Generation Models
  - Moderation Models
  - o1 Models
  - Small Models
  - Transcription Models
  - Web Search Models
- **How to Use**: Click dropdown and select a category to filter available models
- **Purpose**: Filter models by category for easier selection

### **Model Selection Dropdown**
- **Label**: "Select Model"
- **Type**: Dropdown/Selectbox
- **Options**: Changes based on selected category (shows specific model IDs)
- **How to Use**: After selecting a category, choose the specific model you want to use
- **Purpose**: Choose specific model for calculations

### **Model Information Panel**
- **Label**: "📊 Model Information"
- **Type**: Expandable section (expanded by default)
- **Contents**:
  - **Name**: Full model name
  - **Description**: Model capabilities and use cases
  - **Category**: Model category classification
  - **Context Window**: Maximum tokens the model can handle
  - **Pricing Information**: Detailed cost structure showing:
    - Input token costs ($/1M tokens)
    - Output token costs ($/1M tokens)
    - Cached input costs (if supported)
    - Audio token costs (for audio models)
    - Training costs (for fine-tuning models)
    - Per-minute costs (for speech models)
    - Per-character costs (for TTS models)
    - Special pricing notes for tools

### **Pricing Options Section**
- **Header**: "💰 Pricing Options"

#### **Use Cached Input Checkbox**
- **Label**: "Use Cached Input (50% discount)"
- **Type**: Checkbox
- **Default**: Unchecked
- **When Available**: Only enabled for models that support cached input
- **How to Use**: Check this box to apply 50% discount to input token costs
- **Purpose**: Calculate costs when using OpenAI's prompt caching feature

#### **Use Batch API Checkbox**
- **Label**: "Use Batch API (50% discount)"
- **Type**: Checkbox
- **Default**: Unchecked
- **When Available**: Only enabled for models that support batch processing
- **How to Use**: Check this box to apply 50% discount to total costs
- **Purpose**: Calculate costs for non-real-time batch processing workloads

---

## 📑 **MAIN CONTENT TABS**

The application has **7 main tabs**, each with specific functionality:

---

## 📝 **TAB 1: TEXT INPUT**

### **Tab Purpose**: Calculate token counts and costs for text-based inputs

### **Interface varies by selected model type:**

#### **For Standard Text Models (GPT-4o, GPT-3.5, o1, etc.):**

##### **Text Input Area**
- **Label**: "Enter Text for Token Counting"
- **Type**: Multi-line text area
- **Height**: 150 pixels
- **Placeholder**: "Enter your text here..."
- **How to Use**: Type or paste your text content
- **Purpose**: Input text for automatic token calculation

##### **Estimated Output Tokens Input**
- **Label**: "Estimated Output Tokens"
- **Type**: Number input
- **Minimum**: 0
- **Default**: 100
- **Step**: 10
- **How to Use**: Enter expected number of output tokens from the model
- **Purpose**: Calculate total cost including response generation

##### **Results Display (appears after entering text)**
- **Input Tokens Metric**: Shows calculated input tokens
- **Output Tokens Metric**: Shows your estimated output tokens
- **Total Cost Metric**: Shows calculated total cost in USD
- **Context Window Warning**: Appears if total tokens exceed model limits

#### **For Audio Models (audio_tokens/realtime_audio):**

##### **Text Components Section**
- **Text Input Area**: For text portion of audio processing
- **Text Output Tokens**: Number input for expected text response

##### **Audio Components Section**
- **Audio Input Tokens**: Number input for audio input processing
- **Audio Output Tokens**: Number input for audio output generation

##### **Use Cached Input Checkbox**
- **Label**: "Use Cached Input (for supported models)"
- **Purpose**: Apply caching discount for audio models

##### **Results Display (3 columns)**
- **Column 1**: Text Tokens (input + output)
- **Column 2**: Audio Tokens (input + output)
- **Column 3**: Total Cost

#### **For Speech Models (TTS - per_character):**

##### **Text Input Area**
- **Label**: "Text to Convert to Speech"
- **Height**: 150 pixels
- **Placeholder**: "Enter text you want to convert to speech..."

##### **Results Display (3 columns)**
- **Column 1**: Characters count
- **Column 2**: Cost per 1M characters
- **Column 3**: Total Cost

#### **For Transcription Models (per_minute):**

##### **Audio Duration Input**
- **Label**: "Audio Duration (minutes)"
- **Type**: Number input
- **Minimum**: 0.0
- **Default**: 1.0
- **Step**: 0.1

##### **Results Display (2 columns)**
- **Column 1**: Duration in minutes
- **Column 2**: Total Cost

#### **For Embeddings Models:**

##### **Text Input Area**
- **Label**: "Input Text"
- **Purpose**: Text to generate embeddings for

##### **Results Display (3 columns)**
- **Column 1**: Input Tokens
- **Column 2**: Dimensions (model-specific)
- **Column 3**: Total Cost

#### **For Fine-tuning Models:**

##### **Training Section (2 columns)**
- **Column 1**: 
  - Training Hours input (for hour-based pricing)
  - OR Training Tokens input (for token-based pricing)
- **Column 2**:
  - Inference Input Tokens
  - Inference Output Tokens

##### **Data Sharing Checkbox**
- **Label**: "Use Data Sharing (reduced costs)"
- **Purpose**: Apply data sharing discounts where available

##### **Results Display (3 columns)**
- **Column 1**: Training Hours/Tokens
- **Column 2**: Inference Tokens total
- **Column 3**: Total Cost

#### **For Free Models:**

##### **Text Input Area**
- **Standard text input for testing

##### **Results Display (2 columns)**
- **Column 1**: Token count
- **Column 2**: "FREE ✨" indicator

---

## 📁 **TAB 2: FILE UPLOAD**

### **Tab Label**: "📁 File Upload"
### **Purpose**: Process uploaded files and calculate token costs

### **File Uploader Widget**
- **Label**: "Choose files"
- **Type**: File uploader
- **Supported Types**: 'txt', 'pdf', 'docx', 'xlsx', 'xls', 'pptx', 'md', 'py', 'js', 'html', 'css', 'json', 'xml', 'csv'
- **Multiple Files**: Yes
- **Purpose**: Upload documents for text extraction and analysis

### **File Processing Results**
Each uploaded file gets its own expandable section:

#### **File Expander**
- **Label**: "📄 [filename]"
- **State**: Expanded if only one file uploaded
- **Loading Indicator**: "Processing [filename]..." spinner

#### **Text Preview Area**
- **Label**: "Extracted Text Preview"
- **Content**: First 500 characters of extracted text
- **Type**: Text area (read-only)
- **Height**: 150 pixels

#### **Output Tokens Input**
- **Label**: "Estimated Output Tokens for [filename]"
- **Type**: Number input
- **Minimum**: 0
- **Default**: 100
- **Step**: 10
- **Unique Key**: Based on filename

#### **Results Display (3 columns)**
- **Column 1**: Input Tokens from file
- **Column 2**: Estimated Output Tokens
- **Column 3**: Total Cost

#### **Pricing Type Caption**
- **Content**: Shows pricing type (e.g., "Pricing: Standard")

#### **Error Handling**
- **File Processing Errors**: Displayed as error messages
- **Cost Calculation Errors**: Fallback to basic token display

---

## 📊 **TAB 3: ANALYTICS**

### **Tab Label**: "📊 Analytics"
### **Purpose**: Compare model costs and analyze pricing

### **Cost Analytics Header**
- **Label**: "📊 Cost Analytics"

### **Model Cost Comparison Section**
- **Header**: "Model Cost Comparison"

#### **Comparison Text Input**
- **Label**: "Text for Comparison"
- **Type**: Text area
- **Default**: "Hello, how are you today?"
- **Height**: 100 pixels
- **Purpose**: Standard text for comparing across models

#### **Comparison Output Tokens Input**
- **Label**: "Output Tokens for Comparison"
- **Type**: Number input
- **Minimum**: 0
- **Default**: 50
- **Step**: 10

#### **Comparison Results**
- **Interactive Chart**: Plotly bar chart showing cost comparison
- **Data Table**: Pandas DataFrame with model names and costs
- **Models Compared**: Top models (GPT-4o Mini, GPT-4o, o1-mini)

---

## 🔧 **TAB 4: TOOLS & SPECIAL MODELS**

### **Tab Label**: "🔧 Tools & Special Models"
### **Purpose**: Calculate costs for specialized OpenAI services

### **Three-Column Layout**

#### **Column 1: Image Generation**
- **Header**: "🎨 Image Generation"

##### **Image Model Dropdown**
- **Label**: "Image Model"
- **Options**: Available image generation models (DALL-E 3, DALL-E 2)

##### **Number of Images Input**
- **Label**: "Number of Images"
- **Type**: Number input
- **Minimum**: 1
- **Default**: 1

##### **Resolution Dropdown**
- **Label**: "Resolution"
- **Options**: Model-specific resolutions (1024x1024, 1024x1792, etc.)

##### **Quality Dropdown**
- **Label**: "Quality"
- **Options**: standard, hd (varies by resolution)

##### **Cost Display**
- **Metric**: "Total Cost"
- **Format**: Dollar amount with 4 decimal places

##### **Cost Details Expander**
- **Label**: "Image Cost Details"
- **Contents**: Cost per image, resolution, quality settings

#### **Column 2: Web Search**
- **Header**: "🔍 Web Search"

##### **Search Model Dropdown**
- **Label**: "Search Model"
- **Options**: Available web search models

##### **Number of Calls Input**
- **Label**: "Number of Calls"
- **Type**: Number input
- **Minimum**: 1
- **Default**: 1000

##### **Context Size Dropdown**
- **Label**: "Context Size"
- **Options**: small, medium, large (model-specific)

##### **Cost Display**
- **Metric**: "Total Cost"

##### **Search Cost Details Expander**
- **Label**: "Search Cost Details"
- **Contents**: Cost per 1K calls, number of calls, context size

#### **Column 3: Built-in Tools**
- **Header**: "🛠️ Built-in Tools"

##### **Tool Selection Dropdown**
- **Label**: "Tool"
- **Options**: 
  - code_interpreter
  - file_search_storage
  - file_search_tool_call

##### **Dynamic Input Fields (based on tool):**

###### **For Code Interpreter:**
- **Number of Containers Input**
  - **Label**: "Number of Containers"
  - **Minimum**: 1
  - **Default**: 1

###### **For File Search Storage:**
- **GB-Days Input**
  - **Label**: "GB-Days"
  - **Type**: Number input
  - **Minimum**: 0.0
  - **Default**: 1.0
  - **Step**: 0.1

###### **For File Search Tool Call:**
- **Number of Calls Input**
  - **Label**: "Number of Calls"
  - **Minimum**: 1
  - **Default**: 1000

##### **Tool Cost Details Expander**
- **Label**: "Tool Cost Details"
- **Contents**: Tool-specific pricing breakdown

### **Comprehensive Model Summary Section**
- **Header**: "📋 Comprehensive Model Summary"
- **Purpose**: Overview of all available models

#### **Summary Table**
- **Columns**:
  - Model: Model name
  - Category: Model category
  - Pricing Type: Type of pricing structure
  - Base Price: Starting price information
  - Context Window: Token limit
- **Data Sources**:
  - Regular models
  - Image generation models
  - Web search models
  - Built-in tools
- **Display**: Full-width DataFrame

---

## 🔍 **TAB 5: SYSTEM STATUS**

### **Tab Label**: "🔍 System Status & Validation"
### **Purpose**: Monitor system health and validate model data

### **System Validation**
- **Loading Message**: "Running comprehensive system validation..." with spinner

### **Status Overview (4 columns)**
- **Column 1**: 
  - **Metric**: "Pricing Files Loaded"
  - **Format**: "[loaded]/[total]"
- **Column 2**:
  - **Metric**: "Total Models"
  - **Value**: Count of all models
- **Column 3**:
  - **Metric**: "Issues Found"
  - **Value**: Number of validation issues
- **Column 4**:
  - **Metric**: "Special Models"
  - **Value**: Count of special model types

### **Pricing Files Status Section**
- **Header**: "📄 Pricing Files Status"
- **Table Columns**:
  - File: Filename without path
  - Status: "✅ Loaded" or "❌ Missing"
- **Purpose**: Monitor data file availability

### **Models by Category Section**
- **Header**: "📊 Models by Category"
- **Interactive Bar Chart**: Plotly bar chart showing model distribution
- **Data Table**: Category names and model counts

### **Pricing Types Distribution Section**
- **Header**: "💰 Pricing Types Distribution"
- **Interactive Pie Chart**: Plotly pie chart of pricing type distribution
- **Data Table**: Pricing types and counts

### **Issues and Warnings Section**
- **Header**: "⚠️ Issues Found"
- **Content**: Error messages for any validation failures
- **Success State**: "✅ No issues found! All pricing files are properly loaded and validated."

### **System Information Section**
- **Header**: "🖥️ System Information"

#### **Python Libraries Expander**
- **Label**: "📋 Python Libraries"
- **Contents**: Library versions (Streamlit, Pandas, NumPy, Tiktoken)

#### **File Support Expander**
- **Label**: "📋 File Support"
- **Contents**: Supported file processing libraries

---

## 🧪 **TAB 6: MODEL TESTING HUB**

### **Tab Label**: "🧪 Model Testing Hub"
### **Purpose**: Comprehensive testing interface for all model types

### **Testing Hub Header**
- **Title**: "🧪 Comprehensive Model Testing Hub"
- **Subtitle**: "Test and calculate costs for all OpenAI model types with dedicated interfaces."

### **Testing Category Dropdown**
- **Label**: "Choose Testing Category"
- **Options**:
  - "🤖 Text & Chat Models"
  - "🎵 Audio & Speech Models"
  - "🔤 Embeddings Models"
  - "🎨 Image Generation"
  - "🔍 Search & Tools"
  - "🎓 Fine-tuning Models"
  - "🆓 Free Models"

### **Category-Specific Interfaces:**

#### **🤖 Text & Chat Models**
- **Header**: "Text and Chat Models Testing"

##### **Model Selection Dropdown**
- **Label**: "Select Text Model"
- **Options**: Filtered to standard pricing type models

##### **Input Section (2 columns)**
- **Column 1**: Text input area (200px height)
- **Column 2**: 
  - Expected Output Tokens input
  - Use Cached Input checkbox
  - Use Batch API checkbox

##### **Results Display (4 columns)**
- Input Tokens, Output Tokens, Total Tokens, Total Cost

##### **Detailed Cost Breakdown Expander**
- **Label**: "💰 Detailed Cost Breakdown"
- **Contents**: Itemized costs and pricing type

#### **🎵 Audio & Speech Models**
- **Header**: "Audio and Speech Models Testing"

##### **Audio Model Type Radio Buttons**
- **Options**:
  - "🎙️ Audio Processing"
  - "🗣️ Text-to-Speech"
  - "📝 Speech-to-Text"

###### **🎙️ Audio Processing Sub-interface**
- **Model Selection**: Audio/realtime audio models
- **Text Components Section**:
  - Text Input area
  - Text Output Tokens input
- **Audio Components Section**:
  - Audio Input Tokens input
  - Audio Output Tokens input
- **Use Cached Input Checkbox**
- **Results**: Text Tokens, Audio Tokens, Total Cost

###### **🗣️ Text-to-Speech Sub-interface**
- **Model Selection**: per_character pricing models
- **Text Input**: Large text area for TTS conversion
- **Results**: Characters, Cost/1M chars, Total Cost

###### **📝 Speech-to-Text Sub-interface**
- **Model Selection**: per_minute pricing models (Whisper)
- **Audio Duration Input**: Minutes with decimal precision
- **Info**: "Whisper models charge per minute of audio processed"
- **Results**: Duration, Cost/minute, Total Cost

#### **🔤 Embeddings Models**
- **Header**: "Embeddings Models Testing"
- **Model Selection**: Embeddings-specific models
- **Text Input**: Large area for embedding text
- **Use Batch API Checkbox**
- **Info**: "Embeddings only use input tokens"
- **Results**: Input Tokens, Dimensions, Cost/1K tokens, Total Cost

#### **🎨 Image Generation**
- **Header**: "Image Generation Testing"
- **Model Selection**: Image generation models
- **Configuration (3 columns)**:
  - Number of Images (1-10)
  - Resolution dropdown
  - Quality dropdown
- **Results**: Images, Resolution, Cost/Image, Total Cost
- **Sample Prompts Expander**: Pre-written example prompts

#### **🔍 Search & Tools**
- **Header**: "Search and Tools Testing"

##### **Tool Type Radio Buttons**
- **Options**: "🌐 Web Search", "🛠️ Built-in Tools"

###### **🌐 Web Search Sub-interface**
- **Model Selection**: Web search models
- **Configuration**: Number of calls, Context size
- **Results**: Search Calls, Cost/1K Calls, Total Cost

###### **🛠️ Built-in Tools Sub-interface**
- **Tool Selection**: Built-in tools dropdown
- **Dynamic Inputs**: Based on selected tool
- **Results**: Tool-specific metrics and costs

#### **🎓 Fine-tuning Models**
- **Header**: "Fine-tuning Models Testing"
- **Model Selection**: Fine-tuning specific models
- **Info**: "Fine-tuning involves training costs and inference costs"
- **Training Phase Section**: Hours or tokens input
- **Inference Phase Section**: Input/output tokens
- **Checkboxes**: Use Cached Input, Use Data Sharing
- **Results**: Training metrics, Training Cost, Inference Cost, Total Cost

#### **🆓 Free Models**
- **Header**: "Free Models Testing"
- **Model Selection**: Free models (moderation, etc.)
- **Test Input**: Text area for testing
- **Results**: Input Tokens, Model Type, "FREE ✨" indicator
- **Success Message**: "💡 This model is completely free to use!"

### **Quick Testing Tips Expander**
- **Label**: "💡 Quick Testing Tips"
- **Contents**: 
  - Efficient testing strategies for each model type
  - Best practices for different scenarios
  - Cost optimization tips

---

## ⚙️ **TAB 7: MODEL MANAGEMENT**

### **Tab Label**: "⚙️ Model Management"
### **Purpose**: Complete model data management interface

### **Management Header**
- **Title**: "⚙️ Model Management"
- **Subtitle**: "Manage your model data files - add, edit, delete, and maintain model configurations."

### **Management Action Dropdown**
- **Label**: "Choose Management Action"
- **Options**:
  - "📊 View Models"
  - "➕ Add Model"
  - "✏️ Edit Model"
  - "🗑️ Delete Model"
  - "📤 Import/Export"
  - "💾 Backup/Restore"
  - "✅ Validate Data"

### **Management Interfaces by Action:**

#### **📊 View Models**
- **Header**: "📊 Current Models Overview"

##### **Model Statistics (4 columns)**
- **Column 1**: Total Models count
- **Column 2**: Categories count
- **Column 3**: Pricing Types count
- **Column 4**: Last Updated date

##### **Category Selection Dropdown**
- **Label**: "Select Category to View"
- **Options**: All available categories

##### **Models Table**
- **Columns**: Model ID, Name, Category, Pricing Type, Input Cost, Output Cost, Context Window
- **Type**: Full-width DataFrame

##### **Model Details Section**
- **Model Selection Dropdown**: "Select Model for Details"
- **Detailed View Expander**: JSON display of model data
- **Action Buttons (2 columns)**:
  - **Column 1**: "✏️ Edit [model]" button
  - **Column 2**: "🗑️ Delete [model]" button

#### **➕ Add Model**
- **Header**: "➕ Add New Model"

##### **Category Selection**
- **New Category Input**: "New Category (optional)"
- **Existing Category Dropdown**: "Select Existing Category"

##### **Add Model Form**
- **Form Type**: Streamlit form with submit button

###### **Basic Information Fields**
- **Model ID Input**: Required field with placeholder
- **Model Name Input**: Required field
- **Description Text Area**: Required field
- **Display Category Input**: Pre-filled from selection

###### **Pricing Type Dropdown**
- **Options**: standard, audio_tokens, realtime_audio, per_minute, per_character, embeddings, fine_tuning, free, per_image, per_call, tool_specific

###### **Dynamic Form Fields (based on pricing type)**

**For Standard/Embeddings:**
- **Input Cost Number Input** (required)
- **Output Cost Number Input** (required)
- **Cached Input Cost Number Input** (optional)
- **Context Window Number Input** (required)
- **Has Cached Checkbox**
- **Has Batch Checkbox**
- **Dimensions Input** (embeddings only)

**For Audio Models:**
- **Text Input Cost** (required)
- **Text Output Cost** (required)
- **Audio Input Cost** (required)
- **Audio Output Cost** (required)

**For Per-Minute Models:**
- **Cost per Minute Input** (required)

**For Per-Character Models:**
- **Cost per 1M Characters** (required)

**For Free Models:**
- **No additional cost fields**
- **Automatically sets is_free flag**

###### **Submit Button**
- **Label**: "➕ Add Model"
- **Validation**: Checks required fields
- **Success/Error Messages**: Displayed after submission

#### **✏️ Edit Model**
- **Header**: "✏️ Edit Model"

##### **Selection Interface**
- **Category Dropdown**: "Select Category"
- **Model Dropdown**: "Select Model to Edit"

##### **Edit Form**
- **Form Type**: Streamlit form
- **Pre-filled Values**: Current model data
- **Editable Fields**: All model properties
- **Pricing Type**: Display only (non-editable)
- **Submit Button**: "💾 Save Changes"

#### **🗑️ Delete Model**
- **Header**: "🗑️ Delete Model"
- **Warning**: "⚠️ This action cannot be undone without restoring from backup!"

##### **Selection Interface**
- **Category Dropdown**: "Select Category"
- **Model Dropdown**: "Select Model to Delete"

##### **Confirmation Interface**
- **Model Details Display (2 columns)**:
  - **Column 1**: Model ID, Name
  - **Column 2**: Category, Type
- **Confirmation Checkbox**: "I understand that deleting '[model]' is permanent"
- **Delete Button**: "🗑️ Delete Model" (primary type, requires confirmation)

#### **📤 Import/Export**
- **Header**: "📤 Import/Export Models"

##### **Two Sub-tabs**
- **Tab 1**: "📤 Export"
- **Tab 2**: "📥 Import"

###### **📤 Export Tab**
- **Header**: "Export Models"
- **Export All Checkbox**: "Export All Categories"
- **Category Multi-select**: "Select Categories to Export" (if not exporting all)
- **Export Info**: Shows model count and category count
- **Generate Button**: "📤 Generate Export File"
- **Download Button**: "💾 Download Export File" (appears after generation)

###### **📥 Import Tab**
- **Header**: "Import Models"
- **File Uploader**: JSON files only
- **Import Preview**: Table showing Category, Model ID, Name
- **Import Mode Radio**: "Merge (keep existing, add new)" vs "Replace (overwrite categories)"
- **Import Button**: "📥 Import Models"

#### **💾 Backup/Restore**
- **Header**: "💾 Backup & Restore"

##### **Two Sub-tabs**
- **Tab 1**: "💾 Backup"
- **Tab 2**: "🔄 Restore"

###### **💾 Backup Tab**
- **Header**: "Create Backup"
- **Backup Button**: "💾 Create Backup Now"
- **Success/Error Messages**: Status feedback

###### **🔄 Restore Tab**
- **Header**: "Restore from Backup"
- **Available Backups Table**: File, Date, Size columns
- **Backup Selection Dropdown**: "Select Backup to Restore"
- **Warning**: "⚠️ Restoring will overwrite current model data!"
- **Restore Button**: "🔄 Restore Backup"

#### **✅ Validate Data**
- **Header**: "✅ Data Validation"
- **Validation Button**: "🔍 Run Validation"

##### **Validation Results (3 columns)**
- **Column 1**: Total Models count
- **Column 2**: Valid Models count
- **Column 3**: Errors Found count

##### **Results Table**
- **Columns**: Category, Model ID, Status, Message
- **Status Indicators**: "✅ Valid" or "❌ Invalid"

##### **Summary Messages**
- **Success**: "🎉 All models passed validation!"
- **Errors**: "❌ Found [X] validation errors. Please review and fix the issues above."

---

## 💡 **APPLICATION FOOTER**

### **Tips Section**
- **Separator**: Horizontal line
- **Header**: "💡 Tips:"
- **Tip Items**:
  - Use cached input for repeated content to save 50%
  - Consider Batch API for non-real-time workloads to save 50%
  - Different models use different encodings for token counting
  - Image generation and special tools have different pricing structures

---

## 🎯 **HOW TO USE: COMPLETE WORKFLOW**

### **Step 1: Select Your Model**
1. **Choose Category**: Use sidebar dropdown to select model category
2. **Select Model**: Pick specific model from filtered list
3. **Review Info**: Check model information panel for pricing details
4. **Set Options**: Enable cached input or batch API if desired

### **Step 2: Calculate Costs**
**For Text Analysis:**
1. Go to "📝 Text Input" tab
2. Paste your text in the input area
3. Set expected output tokens
4. View automatic cost calculation

**For File Processing:**
1. Go to "📁 File Upload" tab
2. Upload your files (multiple supported)
3. Review extracted text
4. Set output tokens for each file
5. Compare costs across files

**For Specialized Models:**
1. Go to "🔧 Tools & Special Models" tab
2. Configure image generation, web search, or tools
3. View specialized pricing structures

### **Step 3: Compare and Analyze**
1. Use "📊 Analytics" tab for model comparisons
2. Use "🧪 Model Testing Hub" for comprehensive testing
3. Review "🔍 System Status" for validation

### **Step 4: Manage Models (Advanced)**
1. Use "⚙️ Model Management" tab for configuration
2. Add custom models or update pricing
3. Import/export configurations
4. Create backups before changes

---

## 🔧 **CUSTOMIZATION & ADVANCED FEATURES**

### **Model Data Structure**
The application loads models from JSON files in the `pricing_data/` directory:
- `text_models.json`: Standard text models
- `audio_models.json`: Audio processing models
- `embeddings_models.json`: Embedding models
- `fine_tuning_models.json`: Fine-tuning models
- `image_generation_models.json`: Image generation models
- `transcription_models.json`: Speech-to-text models
- `web_search_models.json`: Web search models
- `built_in_tools.json`: OpenAI tools
- `moderation_models.json`: Content moderation models

### **Adding Custom Models**
1. Use the "⚙️ Model Management" tab
2. Select "➕ Add Model"
3. Fill in all required fields
4. Choose appropriate pricing type
5. Save and test

### **Pricing Types Explained**
- **standard**: Input/output token pricing
- **audio_tokens**: Text + audio token pricing
- **realtime_audio**: Real-time audio processing
- **per_minute**: Time-based pricing (Whisper)
- **per_character**: Character-based pricing (TTS)
- **embeddings**: Input-only token pricing
- **fine_tuning**: Training + inference pricing
- **free**: No cost models
- **per_image**: Image generation pricing
- **per_call**: API call pricing
- **tool_specific**: Custom tool pricing

---

## 🚨 **TROUBLESHOOTING**

### **Common Issues**

#### **Application Won't Start**
- Check Python version (3.8+ required)
- Install dependencies: `pip install -r requirements.txt`
- Try: `python run.py` instead of direct Streamlit

#### **Models Not Loading**
- Check `pricing_data/` directory exists
- Verify JSON files are valid
- Use "🔍 System Status" tab to identify issues

#### **File Upload Issues**
- Ensure file format is supported
- Check file size (large files may timeout)
- Verify file isn't corrupted

#### **Incorrect Token Counts**
- Different models use different tokenizers
- Counts are estimates for cost calculation
- Use official OpenAI tokenizer for exact counts

#### **Cost Calculations Seem Wrong**
- Verify model selection is correct
- Check if cached input/batch API options are set properly
- Review cost breakdown for detailed analysis

### **Error Messages**
- **"Model Manager not available"**: Model management system failed to load
- **"Error processing [filename]"**: File format not supported or corrupted
- **"Validation failed"**: Model data doesn't meet requirements
- **"No models found in category"**: Category has no models loaded

---

## 🆕 **FEATURES & UPDATES**

### **Current Version Features**
- 7 comprehensive tabs for different use cases
- Support for all OpenAI model types
- Advanced cost calculation with discounts
- File processing for multiple formats
- Model management system
- Comprehensive validation and testing
- Interactive charts and analytics
- Backup and restore functionality

### **Model Coverage**
- **Text Models**: All GPT variants including o1 models
- **Audio Models**: GPT-4o Audio and Realtime Audio
- **Speech Models**: Whisper transcription and TTS
- **Vision Models**: GPT-4o with vision capabilities
- **Embeddings**: All text-embedding models
- **Fine-tuning**: Custom model training
- **Image Generation**: DALL-E 2 and 3
- **Tools**: Code interpreter, file search, web search
- **Moderation**: Free content moderation

---

## 📞 **SUPPORT & RESOURCES**

### **Getting Help**
1. **System Status**: Check "🔍 System Status" tab first
2. **Validation**: Use "✅ Validate Data" in Model Management
3. **Testing**: Use "🧪 Model Testing Hub" to isolate issues
4. **Backup**: Create backups before making changes

### **Best Practices**
- **Regular Backups**: Use Model Management backup feature
- **Test New Models**: Use Testing Hub before production use
- **Monitor Costs**: Review analytics regularly
- **Keep Updated**: Update pricing data when OpenAI changes rates

### **Technical Details**
- **Framework**: Streamlit for web interface
- **Token Counting**: tiktoken library for accurate counts
- **File Processing**: Multiple libraries for format support
- **Charts**: Plotly for interactive visualizations
- **Data**: JSON-based model configuration

---

## 📄 **LICENSE & ATTRIBUTION**

This application calculates costs based on publicly available OpenAI pricing information. Pricing data should be verified against official OpenAI documentation for production use.

---

**🎉 You now have a complete understanding of every feature, button, dropdown, tab, and option in the OpenAI Token Counter & Cost Calculator!** 