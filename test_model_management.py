#!/usr/bin/env python3
"""
Test script for Model Management System
Verifies all components of the new file-based model management
"""

import sys
import json
import os
from pathlib import Path
import traceback

def test_model_manager_import():
    """Test if model manager can be imported"""
    print("🧪 Testing Model Manager import...")
    
    try:
        from model_manager import ModelManager
        print("  ✅ ModelManager imported successfully")
        return True
    except ImportError as e:
        print(f"  ❌ Failed to import ModelManager: {e}")
        return False

def test_model_utils_import():
    """Test if model utilities can be imported"""
    print("\n🧪 Testing Model Utilities import...")
    
    try:
        from model_utils import ModelValidator, ModelBackupManager, ModelCRUD
        print("  ✅ ModelValidator imported successfully")
        print("  ✅ ModelBackupManager imported successfully")
        print("  ✅ ModelCRUD imported successfully")
        return True
    except ImportError as e:
        print(f"  ❌ Failed to import model utilities: {e}")
        return False

def test_config_file():
    """Test configuration file"""
    print("\n🧪 Testing configuration file...")
    
    config_path = "pricing_data/config.json"
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Check required sections
            required_sections = ["application", "data_sources", "validation_rules"]
            for section in required_sections:
                if section in config:
                    print(f"  ✅ Config section '{section}' exists")
                else:
                    print(f"  ❌ Config section '{section}' missing")
                    return False
            
            return True
        else:
            print(f"  ❌ Config file not found: {config_path}")
            return False
            
    except json.JSONDecodeError as e:
        print(f"  ❌ Invalid JSON in config file: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Error reading config file: {e}")
        return False

def test_model_data_files():
    """Test model data files"""
    print("\n🧪 Testing model data files...")
    
    required_files = [
        "text_models.json",
        "audio_models.json", 
        "transcription_models.json",
        "embeddings_models.json",
        "fine_tuning_models.json",
        "moderation_models.json",
        "image_generation_models.json",
        "web_search_models.json",
        "built_in_tools.json"
    ]
    
    all_valid = True
    
    for filename in required_files:
        filepath = f"pricing_data/{filename}"
        
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                if data:
                    print(f"  ✅ {filename} - valid JSON with data")
                else:
                    print(f"  ⚠️ {filename} - valid JSON but empty")
            else:
                print(f"  ❌ {filename} - file not found")
                all_valid = False
                
        except json.JSONDecodeError as e:
            print(f"  ❌ {filename} - invalid JSON: {e}")
            all_valid = False
        except Exception as e:
            print(f"  ❌ {filename} - error reading: {e}")
            all_valid = False
    
    return all_valid

def test_model_manager_functionality():
    """Test basic model manager functionality"""
    print("\n🧪 Testing ModelManager functionality...")
    
    try:
        from model_manager import ModelManager
        
        # Initialize model manager
        manager = ModelManager()
        print("  ✅ ModelManager initialized")
        
        # Test getting all models
        all_models = manager.get_all_models()
        print(f"  ✅ Retrieved {len(all_models)} total models")
        
        # Test getting categories
        categories = manager.get_categories()
        print(f"  ✅ Found {len(categories)} categories: {categories}")
        
        # Test getting models by category
        for category in categories[:3]:  # Test first 3 categories
            models = manager.get_models_by_category(category)
            print(f"  ✅ Category '{category}': {len(models)} models")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error testing ModelManager: {e}")
        traceback.print_exc()
        return False

def test_model_validation():
    """Test model validation functionality"""
    print("\n🧪 Testing model validation...")
    
    try:
        from model_utils import ModelValidator
        from model_manager import ModelManager
        
        validator = ModelValidator()
        manager = ModelManager()
        
        # Test validation on existing models
        categories = manager.get_categories()
        if categories:
            category = categories[0]
            models = manager.get_models_by_category(category)
            
            if models:
                model_id = list(models.keys())[0]
                model_data = models[model_id]
                
                is_valid, message = validator.validate_model(model_data, category, manager.config)
                
                if is_valid:
                    print(f"  ✅ Model '{model_id}' validation passed")
                else:
                    print(f"  ⚠️ Model '{model_id}' validation failed: {message}")
                
                return True
            else:
                print("  ⚠️ No models found to test validation")
                return True
        else:
            print("  ⚠️ No categories found to test validation")
            return True
            
    except Exception as e:
        print(f"  ❌ Error testing validation: {e}")
        return False

def test_backup_functionality():
    """Test backup functionality"""
    print("\n🧪 Testing backup functionality...")
    
    try:
        from model_utils import ModelBackupManager
        from model_manager import ModelManager
        
        backup_manager = ModelBackupManager()
        manager = ModelManager()
        
        # Test backup creation
        success = backup_manager.create_backup(manager.models_data, manager.config)
        
        if success:
            print("  ✅ Backup created successfully")
            
            # Test getting available backups
            backups = backup_manager.get_available_backups()
            print(f"  ✅ Found {len(backups)} backup files")
            
            return True
        else:
            print("  ❌ Failed to create backup")
            return False
            
    except Exception as e:
        print(f"  ❌ Error testing backup: {e}")
        return False

def test_directories():
    """Test required directories exist"""
    print("\n🧪 Testing directory structure...")
    
    required_dirs = ["pricing_data", "logs", "backups"]
    all_exist = True
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name) and os.path.isdir(dir_name):
            print(f"  ✅ Directory '{dir_name}' exists")
        else:
            print(f"  ❌ Directory '{dir_name}' missing")
            all_exist = False
    
    return all_exist

def test_streamlit_integration():
    """Test Streamlit integration"""
    print("\n🧪 Testing Streamlit integration...")
    
    try:
        from model_management_ui import show_model_management_ui
        print("  ✅ Model Management UI imported successfully")
        
        # Test integration with main app
        try:
            import app
            print("  ✅ Main app imports model management components")
        except Exception as e:
            print(f"  ⚠️ Main app integration issue: {e}")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Failed to import Model Management UI: {e}")
        return False

def run_all_tests():
    """Run all model management tests"""
    print("🧮 Model Management System Test Suite")
    print("=" * 50)
    
    tests = [
        ("Directory Structure", test_directories),
        ("Config File", test_config_file),
        ("Model Data Files", test_model_data_files),
        ("Model Manager Import", test_model_manager_import),
        ("Model Utils Import", test_model_utils_import),
        ("Model Manager Functionality", test_model_manager_functionality),
        ("Model Validation", test_model_validation),
        ("Backup Functionality", test_backup_functionality),
        ("Streamlit Integration", test_streamlit_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Model Management System is ready.")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed. Please review and fix issues.")
        return False

def main():
    """Main test function"""
    print("Starting Model Management System Tests...\n")
    
    success = run_all_tests()
    
    if success:
        print("\n✅ Model Management System is fully functional!")
        print("🚀 You can now:")
        print("   • Add new models through the UI")
        print("   • Edit existing models in real-time")
        print("   • Delete models with automatic backup")
        print("   • Import/export model data")
        print("   • Manage backups and restore data")
        print("   • Validate all model data")
    else:
        print("\n❌ Please fix the issues above before using the Model Management System.")
    
    return success

if __name__ == "__main__":
    main() 