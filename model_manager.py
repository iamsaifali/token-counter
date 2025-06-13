#!/usr/bin/env python3
"""
Model Manager - Comprehensive model data management system
Handles loading, validation, CRUD operations, and backup/restore for all model data
"""

import json
import os
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import traceback

class ModelManager:
    """Centralized model data management system"""
    
    def __init__(self, config_path: str = "pricing_data/config.json"):
        """Initialize the model manager with configuration"""
        self.config_path = config_path
        self.config = self.load_config()
        self.models_data = {}
        self.logger = self.setup_logging()
        self.load_all_models()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from config file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return self.get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if config file doesn't exist"""
        return {
            "data_sources": {
                "models_directory": "pricing_data",
                "model_files": [
                    {"file": "text_models.json", "category": "text_models"},
                    {"file": "audio_models.json", "category": "audio_models"},
                    {"file": "transcription_models.json", "category": "transcription_models"},
                    {"file": "embeddings_models.json", "category": "embeddings_models"},
                    {"file": "fine_tuning_models.json", "category": "fine_tuning_models"},
                    {"file": "moderation_models.json", "category": "moderation_models"},
                    {"file": "image_generation_models.json", "category": "image_generation_models"},
                    {"file": "web_search_models.json", "category": "web_search_models"},
                    {"file": "built_in_tools.json", "category": "built_in_tools"}
                ]
            },
            "validation_rules": {
                "required_fields": {
                    "standard": ["name", "description", "input_cost", "output_cost", "context_window", "category"]
                }
            }
        }
    
    def setup_logging(self) -> logging.Logger:
        """Setup logging for model manager"""
        logger = logging.getLogger('ModelManager')
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create file handler
        handler = logging.FileHandler(log_dir / "model_manager.log")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def load_all_models(self) -> None:
        """Load all model data from configured files"""
        self.models_data = {}
        models_dir = self.config.get("data_sources", {}).get("models_directory", "pricing_data")
        model_files = self.config.get("data_sources", {}).get("model_files", [])
        
        for file_config in model_files:
            file_path = os.path.join(models_dir, file_config["file"])
            category = file_config["category"]
            
            try:
                data = self.load_model_file(file_path)
                if data:
                    self.models_data[category] = data
                    self.logger.info(f"Loaded {len(data)} models from {file_path}")
            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {e}")
                print(f"Warning: Could not load {file_path}: {e}")
    
    def load_model_file(self, file_path: str) -> Dict[str, Any]:
        """Load models from a single JSON file"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Handle nested structure (category -> models)
                    if len(data) == 1:
                        return list(data.values())[0]
                    return data
            return {}
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return {}
    
    def save_model_file(self, file_path: str, data: Dict[str, Any], category_name: str) -> bool:
        """Save models to a JSON file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Wrap data in category structure for consistency
            output_data = {category_name: data}
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved {len(data)} models to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving {file_path}: {e}")
            return False
    
    def get_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all models across all categories"""
        all_models = {}
        for category_data in self.models_data.values():
            all_models.update(category_data)
        return all_models
    
    def get_models_by_category(self, category: str) -> Dict[str, Any]:
        """Get models for a specific category"""
        return self.models_data.get(category, {})
    
    def get_categories(self) -> List[str]:
        """Get list of all available categories"""
        return list(self.models_data.keys())
    
    def get_file_for_category(self, category: str) -> Optional[str]:
        """Get the file path for a given category"""
        model_files = self.config.get("data_sources", {}).get("model_files", [])
        models_dir = self.config.get("data_sources", {}).get("models_directory", "pricing_data")
        
        for file_config in model_files:
            if file_config["category"] == category:
                return os.path.join(models_dir, file_config["file"])
        return None
    
    def add_model(self, category: str, model_id: str, model_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Add a new model to the specified category"""
        try:
            # Validate model data
            validation_result = self.validate_model(model_data, category)
            if not validation_result[0]:
                return False, f"Validation failed: {validation_result[1]}"
            
            # Get category data
            if category not in self.models_data:
                self.models_data[category] = {}
            
            # Check if model already exists
            if model_id in self.models_data[category]:
                return False, f"Model '{model_id}' already exists in category '{category}'"
            
            # Add model
            self.models_data[category][model_id] = model_data
            
            # Save to file
            file_path = self.get_file_for_category(category)
            if file_path:
                category_name = category.replace("_models", "").replace("_", " ").title() + " Models"
                if self.save_model_file(file_path, self.models_data[category], category_name):
                    self.logger.info(f"Added model '{model_id}' to category '{category}'")
                    return True, "Model added successfully"
                else:
                    return False, "Failed to save model file"
            else:
                return False, f"No file configured for category '{category}'"
                
        except Exception as e:
            self.logger.error(f"Error adding model: {e}")
            return False, f"Error adding model: {str(e)}"
    
    def update_model(self, category: str, model_id: str, model_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Update an existing model"""
        try:
            # Check if model exists
            if category not in self.models_data or model_id not in self.models_data[category]:
                return False, f"Model '{model_id}' not found in category '{category}'"
            
            # Validate model data
            validation_result = self.validate_model(model_data, category)
            if not validation_result[0]:
                return False, f"Validation failed: {validation_result[1]}"
            
            # Create backup
            self.create_backup()
            
            # Update model
            self.models_data[category][model_id] = model_data
            
            # Save to file
            file_path = self.get_file_for_category(category)
            if file_path:
                category_name = category.replace("_models", "").replace("_", " ").title() + " Models"
                if self.save_model_file(file_path, self.models_data[category], category_name):
                    self.logger.info(f"Updated model '{model_id}' in category '{category}'")
                    return True, "Model updated successfully"
                else:
                    return False, "Failed to save model file"
            else:
                return False, f"No file configured for category '{category}'"
                
        except Exception as e:
            self.logger.error(f"Error updating model: {e}")
            return False, f"Error updating model: {str(e)}"
    
    def delete_model(self, category: str, model_id: str) -> Tuple[bool, str]:
        """Delete a model from the specified category"""
        try:
            # Check if model exists
            if category not in self.models_data or model_id not in self.models_data[category]:
                return False, f"Model '{model_id}' not found in category '{category}'"
            
            # Create backup
            self.create_backup()
            
            # Delete model
            del self.models_data[category][model_id]
            
            # Save to file
            file_path = self.get_file_for_category(category)
            if file_path:
                category_name = category.replace("_models", "").replace("_", " ").title() + " Models"
                if self.save_model_file(file_path, self.models_data[category], category_name):
                    self.logger.info(f"Deleted model '{model_id}' from category '{category}'")
                    return True, "Model deleted successfully"
                else:
                    return False, "Failed to save model file"
            else:
                return False, f"No file configured for category '{category}'"
                
        except Exception as e:
            self.logger.error(f"Error deleting model: {e}")
            return False, f"Error deleting model: {str(e)}"
    
    def validate_model(self, model_data: Dict[str, Any], category: str) -> Tuple[bool, str]:
        """Validate model data according to configuration rules"""
        try:
            # Get pricing type from model data or infer from category
            pricing_type = model_data.get('pricing_type', 'standard')
            if category == 'audio_models':
                pricing_type = 'audio_tokens'
            elif category == 'transcription_models':
                pricing_type = 'per_minute'
            elif category == 'embeddings_models':
                pricing_type = 'embeddings'
            elif category == 'fine_tuning_models':
                pricing_type = 'fine_tuning'
            elif category == 'moderation_models':
                pricing_type = 'free'
            elif category == 'image_generation_models':
                pricing_type = 'per_image'
            elif category == 'web_search_models':
                pricing_type = 'per_call'
            elif category == 'built_in_tools':
                pricing_type = 'tool_specific'
            
            # Get required fields for this pricing type
            validation_rules = self.config.get("validation_rules", {})
            required_fields = validation_rules.get("required_fields", {}).get(pricing_type, [])
            
            # Check required fields
            missing_fields = []
            for field in required_fields:
                if field not in model_data:
                    missing_fields.append(field)
            
            if missing_fields:
                return False, f"Missing required fields: {', '.join(missing_fields)}"
            
            # Validate field types
            field_types = validation_rules.get("field_types", {})
            for field, expected_type in field_types.items():
                if field in model_data:
                    value = model_data[field]
                    if expected_type == "string" and not isinstance(value, str):
                        return False, f"Field '{field}' must be a string"
                    elif expected_type == "number" and not isinstance(value, (int, float)):
                        return False, f"Field '{field}' must be a number"
                    elif expected_type == "integer" and not isinstance(value, int):
                        return False, f"Field '{field}' must be an integer"
                    elif expected_type == "boolean" and not isinstance(value, bool):
                        return False, f"Field '{field}' must be a boolean"
            
            # Validate cost ranges
            cost_ranges = validation_rules.get("cost_ranges", {})
            for cost_field in ["input_cost", "output_cost", "cached_input_cost", "cost_per_minute"]:
                if cost_field in model_data:
                    cost = model_data[cost_field]
                    if isinstance(cost, (int, float)):
                        min_cost = cost_ranges.get("min_cost", 0.0)
                        max_cost = cost_ranges.get("max_cost", 1000.0)
                        if cost < min_cost or cost > max_cost:
                            return False, f"Cost field '{cost_field}' must be between {min_cost} and {max_cost}"
            
            return True, "Validation passed"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def create_backup(self) -> bool:
        """Create a backup of all model data"""
        try:
            backup_dir = Path(self.config.get("backup_settings", {}).get("backup_directory", "backups"))
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"models_backup_{timestamp}.json"
            
            backup_data = {
                "timestamp": timestamp,
                "config": self.config,
                "models_data": self.models_data
            }
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Created backup: {backup_path}")
            
            # Clean up old backups
            self.cleanup_old_backups()
            
            return True
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            return False
    
    def cleanup_old_backups(self) -> None:
        """Remove old backup files according to configuration"""
        try:
            backup_dir = Path(self.config.get("backup_settings", {}).get("backup_directory", "backups"))
            max_backups = self.config.get("backup_settings", {}).get("max_backups", 30)
            
            if backup_dir.exists():
                backup_files = sorted(backup_dir.glob("models_backup_*.json"))
                if len(backup_files) > max_backups:
                    for old_backup in backup_files[:-max_backups]:
                        old_backup.unlink()
                        self.logger.info(f"Removed old backup: {old_backup}")
        except Exception as e:
            self.logger.error(f"Error cleaning up backups: {e}")
    
    def restore_backup(self, backup_path: str) -> Tuple[bool, str]:
        """Restore models from a backup file"""
        try:
            if not os.path.exists(backup_path):
                return False, f"Backup file not found: {backup_path}"
            
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            # Validate backup structure
            if "models_data" not in backup_data:
                return False, "Invalid backup file format"
            
            # Create current backup before restore
            self.create_backup()
            
            # Restore data
            self.models_data = backup_data["models_data"]
            
            # Save all model files
            models_dir = self.config.get("data_sources", {}).get("models_directory", "pricing_data")
            model_files = self.config.get("data_sources", {}).get("model_files", [])
            
            for file_config in model_files:
                category = file_config["category"]
                if category in self.models_data:
                    file_path = os.path.join(models_dir, file_config["file"])
                    category_name = category.replace("_models", "").replace("_", " ").title() + " Models"
                    self.save_model_file(file_path, self.models_data[category], category_name)
            
            self.logger.info(f"Restored backup from: {backup_path}")
            return True, "Backup restored successfully"
            
        except Exception as e:
            self.logger.error(f"Error restoring backup: {e}")
            return False, f"Error restoring backup: {str(e)}"
    
    def get_available_backups(self) -> List[Dict[str, Any]]:
        """Get list of available backup files"""
        try:
            backup_dir = Path(self.config.get("backup_settings", {}).get("backup_directory", "backups"))
            backups = []
            
            if backup_dir.exists():
                for backup_file in sorted(backup_dir.glob("models_backup_*.json"), reverse=True):
                    try:
                        stat = backup_file.stat()
                        timestamp_str = backup_file.stem.replace("models_backup_", "")
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        
                        backups.append({
                            "file": str(backup_file),
                            "name": backup_file.name,
                            "timestamp": timestamp,
                            "size": stat.st_size,
                            "formatted_time": timestamp.strftime("%Y-%m-%d %H:%M:%S")
                        })
                    except Exception as e:
                        self.logger.warning(f"Error reading backup file {backup_file}: {e}")
            
            return backups
        except Exception as e:
            self.logger.error(f"Error getting backups: {e}")
            return []
    
    def export_models(self, file_path: str, categories: Optional[List[str]] = None) -> Tuple[bool, str]:
        """Export models to a file"""
        try:
            export_data = {}
            
            if categories:
                for category in categories:
                    if category in self.models_data:
                        export_data[category] = self.models_data[category]
            else:
                export_data = self.models_data.copy()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Exported models to: {file_path}")
            return True, "Models exported successfully"
            
        except Exception as e:
            self.logger.error(f"Error exporting models: {e}")
            return False, f"Error exporting models: {str(e)}"
    
    def import_models(self, file_path: str, merge: bool = True) -> Tuple[bool, str]:
        """Import models from a file"""
        try:
            if not os.path.exists(file_path):
                return False, f"Import file not found: {file_path}"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Create backup before import
            self.create_backup()
            
            imported_count = 0
            errors = []
            
            for category, models in import_data.items():
                if not merge:
                    self.models_data[category] = {}
                
                if category not in self.models_data:
                    self.models_data[category] = {}
                
                for model_id, model_data in models.items():
                    # Validate model
                    validation_result = self.validate_model(model_data, category)
                    if validation_result[0]:
                        self.models_data[category][model_id] = model_data
                        imported_count += 1
                    else:
                        errors.append(f"Model '{model_id}' in category '{category}': {validation_result[1]}")
            
            # Save all updated files
            models_dir = self.config.get("data_sources", {}).get("models_directory", "pricing_data")
            model_files = self.config.get("data_sources", {}).get("model_files", [])
            
            for file_config in model_files:
                category = file_config["category"]
                if category in import_data:
                    file_path_save = os.path.join(models_dir, file_config["file"])
                    category_name = category.replace("_models", "").replace("_", " ").title() + " Models"
                    self.save_model_file(file_path_save, self.models_data[category], category_name)
            
            result_message = f"Imported {imported_count} models"
            if errors:
                result_message += f". {len(errors)} errors occurred."
            
            self.logger.info(f"Imported models from: {file_path}. {result_message}")
            return True, result_message
            
        except Exception as e:
            self.logger.error(f"Error importing models: {e}")
            return False, f"Error importing models: {str(e)}"
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get statistics about the model data"""
        stats = {
            "total_models": 0,
            "categories": len(self.models_data),
            "category_breakdown": {},
            "pricing_types": set(),
            "last_updated": datetime.now().isoformat()
        }
        
        for category, models in self.models_data.items():
            count = len(models)
            stats["total_models"] += count
            stats["category_breakdown"][category] = count
            
            # Collect pricing types
            for model in models.values():
                pricing_type = model.get("pricing_type", "standard")
                stats["pricing_types"].add(pricing_type)
        
        stats["pricing_types"] = list(stats["pricing_types"])
        return stats 