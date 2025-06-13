#!/usr/bin/env python3
"""
Model Utilities - Helper functions for model management
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

class ModelValidator:
    """Model data validation utilities"""
    
    @staticmethod
    def validate_model(model_data: Dict[str, Any], category: str, config: Dict[str, Any]) -> Tuple[bool, str]:
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
            
            # Get required fields for this pricing type
            validation_rules = config.get("validation_rules", {})
            required_fields = validation_rules.get("required_fields", {}).get(pricing_type, 
                ["name", "description", "category"])
            
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
            
            return True, "Validation passed"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

class ModelBackupManager:
    """Backup and restore utilities for model data"""
    
    def __init__(self, backup_dir: str = "backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
    
    def create_backup(self, models_data: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """Create a backup of all model data"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"models_backup_{timestamp}.json"
            
            backup_data = {
                "timestamp": timestamp,
                "config": config,
                "models_data": models_data
            }
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            # Clean up old backups
            self.cleanup_old_backups()
            
            return True
        except Exception as e:
            print(f"Error creating backup: {e}")
            return False
    
    def cleanup_old_backups(self, max_backups: int = 30) -> None:
        """Remove old backup files"""
        try:
            backup_files = sorted(self.backup_dir.glob("models_backup_*.json"))
            if len(backup_files) > max_backups:
                for old_backup in backup_files[:-max_backups]:
                    old_backup.unlink()
        except Exception as e:
            print(f"Error cleaning up backups: {e}")
    
    def get_available_backups(self) -> List[Dict[str, Any]]:
        """Get list of available backup files"""
        try:
            backups = []
            
            for backup_file in sorted(self.backup_dir.glob("models_backup_*.json"), reverse=True):
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
                    print(f"Error reading backup file {backup_file}: {e}")
            
            return backups
        except Exception as e:
            print(f"Error getting backups: {e}")
            return []

class ModelCRUD:
    """CRUD operations for model management"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.validator = ModelValidator()
        self.backup_manager = ModelBackupManager()
    
    def add_model(self, category: str, model_id: str, model_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Add a new model to the specified category"""
        try:
            # Validate model data
            validation_result = self.validator.validate_model(model_data, category, self.model_manager.config)
            if not validation_result[0]:
                return False, f"Validation failed: {validation_result[1]}"
            
            # Get category data
            if category not in self.model_manager.models_data:
                self.model_manager.models_data[category] = {}
            
            # Check if model already exists
            if model_id in self.model_manager.models_data[category]:
                return False, f"Model '{model_id}' already exists in category '{category}'"
            
            # Add model
            self.model_manager.models_data[category][model_id] = model_data
            
            # Save to file
            file_path = self.get_file_for_category(category)
            if file_path:
                category_name = self.format_category_name(category)
                if self.model_manager.save_model_file(file_path, self.model_manager.models_data[category], category_name):
                    self.model_manager.logger.info(f"Added model '{model_id}' to category '{category}'")
                    return True, "Model added successfully"
                else:
                    return False, "Failed to save model file"
            else:
                return False, f"No file configured for category '{category}'"
                
        except Exception as e:
            return False, f"Error adding model: {str(e)}"
    
    def update_model(self, category: str, model_id: str, model_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Update an existing model"""
        try:
            # Check if model exists
            if category not in self.model_manager.models_data or model_id not in self.model_manager.models_data[category]:
                return False, f"Model '{model_id}' not found in category '{category}'"
            
            # Validate model data
            validation_result = self.validator.validate_model(model_data, category, self.model_manager.config)
            if not validation_result[0]:
                return False, f"Validation failed: {validation_result[1]}"
            
            # Create backup
            self.backup_manager.create_backup(self.model_manager.models_data, self.model_manager.config)
            
            # Update model
            self.model_manager.models_data[category][model_id] = model_data
            
            # Save to file
            file_path = self.get_file_for_category(category)
            if file_path:
                category_name = self.format_category_name(category)
                if self.model_manager.save_model_file(file_path, self.model_manager.models_data[category], category_name):
                    return True, "Model updated successfully"
                else:
                    return False, "Failed to save model file"
            else:
                return False, f"No file configured for category '{category}'"
                
        except Exception as e:
            return False, f"Error updating model: {str(e)}"
    
    def delete_model(self, category: str, model_id: str) -> Tuple[bool, str]:
        """Delete a model from the specified category"""
        try:
            # Check if model exists
            if category not in self.model_manager.models_data or model_id not in self.model_manager.models_data[category]:
                return False, f"Model '{model_id}' not found in category '{category}'"
            
            # Create backup
            self.backup_manager.create_backup(self.model_manager.models_data, self.model_manager.config)
            
            # Delete model
            del self.model_manager.models_data[category][model_id]
            
            # Save to file
            file_path = self.get_file_for_category(category)
            if file_path:
                category_name = self.format_category_name(category)
                if self.model_manager.save_model_file(file_path, self.model_manager.models_data[category], category_name):
                    return True, "Model deleted successfully"
                else:
                    return False, "Failed to save model file"
            else:
                return False, f"No file configured for category '{category}'"
                
        except Exception as e:
            return False, f"Error deleting model: {str(e)}"
    
    def get_file_for_category(self, category: str) -> Optional[str]:
        """Get the file path for a given category"""
        model_files = self.model_manager.config.get("data_sources", {}).get("model_files", [])
        models_dir = self.model_manager.config.get("data_sources", {}).get("models_directory", "pricing_data")
        
        for file_config in model_files:
            if file_config["category"] == category:
                return os.path.join(models_dir, file_config["file"])
        return None
    
    def format_category_name(self, category: str) -> str:
        """Format category name for file storage"""
        return category.replace("_models", "").replace("_", " ").title() + " Models" 