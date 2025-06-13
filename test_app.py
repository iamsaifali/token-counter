#!/usr/bin/env python3
"""
Test script for OpenAI Token Counter & Cost Calculator
Verifies that all components work correctly before running the main application.
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    
    required_modules = [
        'streamlit',
        'tiktoken',
        'pandas',
        'numpy',
        'docx',
        'PyPDF2',
        'pdfplumber',
        'openpyxl',
        'xlrd',
        'pptx',
        'chardet',
        'PIL',
        'plotly'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("📦 Please install missing dependencies with: pip install -r requirements.txt")
        return False
    
    print("✅ All imports successful!")
    return True

def test_tiktoken():
    """Test tiktoken functionality"""
    print("\n🧪 Testing tiktoken...")
    
    try:
        import tiktoken
        
        # Test different encodings
        encodings_to_test = [
            ("cl100k_base", "Hello, world!"),
            ("o200k_base", "Hello, world!"),
        ]
        
        for encoding_name, test_text in encodings_to_test:
            try:
                encoding = tiktoken.get_encoding(encoding_name)
                tokens = encoding.encode(test_text)
                decoded = encoding.decode(tokens)
                
                print(f"  ✅ {encoding_name}: {len(tokens)} tokens")
                assert decoded == test_text, "Encoding/decoding mismatch"
                
            except Exception as e:
                print(f"  ❌ {encoding_name}: {e}")
                return False
        
        print("✅ tiktoken working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ tiktoken test failed: {e}")
        return False

def test_file_processing():
    """Test file processing utilities"""
    print("\n🧪 Testing file processing...")
    
    try:
        # Test text analysis
        test_text = "This is a test document. It has multiple sentences. Let's see how it processes."
        
        # Import our modules
        sys.path.append('.')
        from app import TokenCalculator
        
        # Test token counting functionality
        tokens = TokenCalculator.count_tokens(test_text, "gpt-4o")
        print(f"  ✅ Token counting: {tokens} tokens for test text")
        
        # Test text length calculations
        chars = len(test_text)
        words = len(test_text.split())
        print(f"  ✅ Text analysis: {chars} characters, {words} words")
        
        print("✅ File processing tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ File processing test failed: {e}")
        traceback.print_exc()
        return False

def test_models_data():
    """Test OpenAI models data"""
    print("\n🧪 Testing models data...")
    
    try:
        sys.path.append('.')
        from app import OpenAIModels
        
        # Test getting all models
        models = OpenAIModels.get_all_models()
        print(f"  ✅ Loaded {len(models)} text models")
        
        # Test specific model
        if "gpt-4o-2024-11-20" in models:
            model = models["gpt-4o-2024-11-20"]
            required_fields = ['name', 'description', 'input_cost', 'output_cost']
            
            for field in required_fields:
                if field not in model:
                    print(f"  ❌ Missing field '{field}' in model data")
                    return False
            
            print(f"  ✅ Model structure validated: {model['name']}")
        
        # Test image models
        image_models = OpenAIModels.get_image_models()
        print(f"  ✅ Loaded {len(image_models)} image models")
        
        # Test embedding models
        embedding_models = OpenAIModels.get_embedding_models()
        print(f"  ✅ Loaded {len(embedding_models)} embedding models")
        
        print("✅ Models data tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Models data test failed: {e}")
        traceback.print_exc()
        return False

def test_token_calculator():
    """Test token calculator functionality"""
    print("\n🧪 Testing token calculator...")
    
    try:
        sys.path.append('.')
        from app import TokenCalculator, OpenAIModels
        
        test_text = "This is a test for token counting. Let's see how many tokens this generates."
        
        # Test token counting
        token_count = TokenCalculator.count_tokens(test_text, "gpt-4o")
        print(f"  ✅ Token counting: {token_count} tokens for test text")
        
        # Test cost calculation
        models = OpenAIModels.get_all_models()
        model_info = models["gpt-4o-2024-11-20"]
        
        total_cost, breakdown = TokenCalculator.calculate_cost(
            input_tokens=token_count,
            output_tokens=50,
            model_info=model_info
        )
        
        print(f"  ✅ Cost calculation: ${total_cost:.6f}")
        print(f"  ✅ Cost breakdown keys: {list(breakdown.keys())}")
        
        print("✅ Token calculator tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Token calculator test failed: {e}")
        traceback.print_exc()
        return False

def test_streamlit_components():
    """Test Streamlit-specific components"""
    print("\n🧪 Testing Streamlit components...")
    
    try:
        import streamlit as st
        print(f"  ✅ Streamlit version: {st.__version__}")
        
        # Test plotly
        import plotly.graph_objects as go
        fig = go.Figure()
        print(f"  ✅ Plotly working")
        
        # Test pandas
        import pandas as pd
        df = pd.DataFrame({'test': [1, 2, 3]})
        print(f"  ✅ Pandas working")
        
        print("✅ Streamlit components tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Streamlit components test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🧮 OpenAI Token Counter - System Tests")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Tiktoken Tests", test_tiktoken),
        ("File Processing Tests", test_file_processing),
        ("Models Data Tests", test_models_data),
        ("Token Calculator Tests", test_token_calculator),
        ("Streamlit Components Tests", test_streamlit_components),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The application should work correctly.")
        print("🚀 You can now run: python run.py")
        return True
    else:
        print("❌ Some tests failed. Please fix the issues before running the application.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 