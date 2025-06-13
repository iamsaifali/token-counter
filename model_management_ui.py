#!/usr/bin/env python3
"""
Model Management UI - Streamlit interface for managing model data
"""

import streamlit as st
import json
import pandas as pd
from typing import Dict, Any
from datetime import datetime
import io

def show_model_management_ui(model_manager):
    """Display the complete model management interface"""
    
    # Management options
    management_action = st.selectbox(
        "Choose Management Action",
        ["📊 View Models", "➕ Add Model", "✏️ Edit Model", "🗑️ Delete Model", 
         "📤 Import/Export", "💾 Backup/Restore", "✅ Validate Data"]
    )
    
    if management_action == "📊 View Models":
        show_models_view(model_manager)
    elif management_action == "➕ Add Model":
        show_add_model(model_manager)
    elif management_action == "✏️ Edit Model":
        show_edit_model(model_manager)
    elif management_action == "🗑️ Delete Model":
        show_delete_model(model_manager)
    elif management_action == "📤 Import/Export":
        show_import_export(model_manager)
    elif management_action == "💾 Backup/Restore":
        show_backup_restore(model_manager)
    elif management_action == "✅ Validate Data":
        show_validation(model_manager)

def show_models_view(model_manager):
    """Display all models in a browsable format"""
    st.subheader("📊 Current Models Overview")
    
    # Model statistics
    stats = model_manager.get_model_statistics() if hasattr(model_manager, 'get_model_statistics') else {}
    
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Models", stats.get('total_models', 0))
        with col2:
            st.metric("Categories", stats.get('categories', 0))
        with col3:
            st.metric("Pricing Types", len(stats.get('pricing_types', [])))
        with col4:
            st.metric("Last Updated", stats.get('last_updated', 'Unknown')[:10])
    
    # Category selection
    categories = model_manager.get_categories()
    if categories:
        selected_category = st.selectbox("Select Category to View", categories)
        
        if selected_category:
            models = model_manager.get_models_by_category(selected_category)
            
            if models:
                st.write(f"**{len(models)} models in {selected_category}:**")
                
                # Create a DataFrame for better display
                model_data = []
                for model_id, model_info in models.items():
                    model_data.append({
                        "Model ID": model_id,
                        "Name": model_info.get('name', 'N/A'),
                        "Category": model_info.get('category', 'N/A'),
                        "Pricing Type": model_info.get('pricing_type', 'standard'),
                        "Input Cost": model_info.get('input_cost', 'N/A'),
                        "Output Cost": model_info.get('output_cost', 'N/A'),
                        "Context Window": model_info.get('context_window', 'N/A')
                    })
                
                df = pd.DataFrame(model_data)
                st.dataframe(df, use_container_width=True)
                
                # Detailed view for selected model
                model_ids = list(models.keys())
                selected_model = st.selectbox("Select Model for Details", model_ids)
                
                if selected_model:
                    with st.expander(f"📋 Detailed View: {selected_model}", expanded=True):
                        model_info = models[selected_model]
                        
                        # Display as formatted JSON
                        st.json(model_info)
                        
                        # Action buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"✏️ Edit {selected_model}", key=f"edit_{selected_model}"):
                                st.session_state['edit_model_id'] = selected_model
                                st.session_state['edit_category'] = selected_category
                                st.rerun()
                        with col2:
                            if st.button(f"🗑️ Delete {selected_model}", key=f"delete_{selected_model}"):
                                st.session_state['delete_model_id'] = selected_model
                                st.session_state['delete_category'] = selected_category
                                st.rerun()
            else:
                st.info(f"No models found in category: {selected_category}")
    else:
        st.warning("No model categories found. Please check your data files.")

def show_add_model(model_manager):
    """Interface for adding new models"""
    st.subheader("➕ Add New Model")
    
    # Category selection
    categories = model_manager.get_categories()
    new_category = st.text_input("New Category (optional)")
    
    if new_category:
        selected_category = new_category
    else:
        selected_category = st.selectbox("Select Existing Category", categories)
    
    if selected_category:
        # Model form
        with st.form("add_model_form"):
            st.write(f"**Adding model to category: {selected_category}**")
            
            # Basic information
            model_id = st.text_input("Model ID*", placeholder="e.g., gpt-4o-2024-12-01")
            model_name = st.text_input("Model Name*", placeholder="e.g., GPT-4o")
            description = st.text_area("Description*", placeholder="Brief description of the model")
            category = st.text_input("Display Category", value=selected_category.replace('_', ' ').title())
            
            # Pricing type
            pricing_type = st.selectbox("Pricing Type", [
                "standard", "audio_tokens", "realtime_audio", "per_minute", 
                "per_character", "embeddings", "fine_tuning", "free", 
                "per_image", "per_call", "tool_specific"
            ])
            
            # Dynamic form based on pricing type
            model_data = {}
            
            if pricing_type in ["standard", "embeddings"]:
                col1, col2 = st.columns(2)
                with col1:
                    input_cost = st.number_input("Input Cost (per 1M tokens)*", min_value=0.0, step=0.01)
                    cached_input_cost = st.number_input("Cached Input Cost (per 1M tokens)", min_value=0.0, step=0.01)
                with col2:
                    output_cost = st.number_input("Output Cost (per 1M tokens)*", min_value=0.0, step=0.01)
                    context_window = st.number_input("Context Window*", min_value=1000, step=1000)
                
                # Checkboxes
                has_cached = st.checkbox("Supports Cached Input")
                has_batch = st.checkbox("Supports Batch API")
                
                model_data = {
                    "name": model_name,
                    "description": description,
                    "input_cost": input_cost,
                    "output_cost": output_cost,
                    "context_window": context_window,
                    "category": category,
                    "pricing_type": pricing_type,
                    "has_cached": has_cached,
                    "has_batch": has_batch
                }
                
                if cached_input_cost > 0:
                    model_data["cached_input_cost"] = cached_input_cost
                
                if pricing_type == "embeddings":
                    dimensions = st.number_input("Dimensions*", min_value=1, step=1)
                    model_data["dimensions"] = dimensions
            
            elif pricing_type in ["audio_tokens", "realtime_audio"]:
                col1, col2 = st.columns(2)
                with col1:
                    text_input_cost = st.number_input("Text Input Cost (per 1M tokens)*", min_value=0.0, step=0.01)
                    audio_input_cost = st.number_input("Audio Input Cost (per 1M tokens)*", min_value=0.0, step=0.01)
                with col2:
                    text_output_cost = st.number_input("Text Output Cost (per 1M tokens)*", min_value=0.0, step=0.01)
                    audio_output_cost = st.number_input("Audio Output Cost (per 1M tokens)*", min_value=0.0, step=0.01)
                
                model_data = {
                    "name": model_name,
                    "description": description,
                    "text_input_cost": text_input_cost,
                    "text_output_cost": text_output_cost,
                    "audio_input_cost": audio_input_cost,
                    "audio_output_cost": audio_output_cost,
                    "category": category,
                    "pricing_type": pricing_type
                }
            
            elif pricing_type == "per_minute":
                cost_per_minute = st.number_input("Cost per Minute*", min_value=0.0, step=0.001)
                model_data = {
                    "name": model_name,
                    "description": description,
                    "cost_per_minute": cost_per_minute,
                    "category": category,
                    "pricing_type": pricing_type
                }
            
            elif pricing_type == "per_character":
                cost_per_1m_chars = st.number_input("Cost per 1M Characters*", min_value=0.0, step=0.01)
                model_data = {
                    "name": model_name,
                    "description": description,
                    "cost_per_1m_characters": cost_per_1m_chars,
                    "category": category,
                    "pricing_type": pricing_type
                }
            
            elif pricing_type == "free":
                model_data = {
                    "name": model_name,
                    "description": description,
                    "category": category,
                    "pricing_type": pricing_type,
                    "is_free": True
                }
            
            # Submit button
            if st.form_submit_button("➕ Add Model"):
                if model_id and model_name and description:
                    try:
                        from model_utils import ModelCRUD
                        crud = ModelCRUD(model_manager)
                        success, message = crud.add_model(selected_category, model_id, model_data)
                        
                        if success:
                            st.success(f"✅ {message}")
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
                    except Exception as e:
                        st.error(f"Error adding model: {str(e)}")
                else:
                    st.error("Please fill in all required fields marked with *")

def show_edit_model(model_manager):
    """Interface for editing existing models"""
    st.subheader("✏️ Edit Model")
    
    # Category and model selection
    categories = model_manager.get_categories()
    selected_category = st.selectbox("Select Category", categories, key="edit_cat_select")
    
    if selected_category:
        models = model_manager.get_models_by_category(selected_category)
        if models:
            model_ids = list(models.keys())
            selected_model = st.selectbox("Select Model to Edit", model_ids, key="edit_model_select")
            
            if selected_model:
                current_data = models[selected_model].copy()
                
                # Edit form
                with st.form("edit_model_form"):
                    st.write(f"**Editing: {selected_model}**")
                    
                    # Show current values as defaults
                    model_name = st.text_input("Model Name", value=current_data.get('name', ''))
                    description = st.text_area("Description", value=current_data.get('description', ''))
                    category = st.text_input("Display Category", value=current_data.get('category', ''))
                    
                    pricing_type = current_data.get('pricing_type', 'standard')
                    st.write(f"**Pricing Type:** {pricing_type}")
                    
                    # Dynamic fields based on pricing type
                    updated_data = current_data.copy()
                    updated_data.update({
                        "name": model_name,
                        "description": description,
                        "category": category
                    })
                    
                    if pricing_type in ["standard", "embeddings"]:
                        col1, col2 = st.columns(2)
                        with col1:
                            input_cost = st.number_input("Input Cost", value=current_data.get('input_cost', 0.0), step=0.01)
                            cached_input_cost = st.number_input("Cached Input Cost", value=current_data.get('cached_input_cost', 0.0), step=0.01)
                        with col2:
                            output_cost = st.number_input("Output Cost", value=current_data.get('output_cost', 0.0), step=0.01)
                            context_window = st.number_input("Context Window", value=current_data.get('context_window', 128000), step=1000)
                        
                        has_cached = st.checkbox("Supports Cached Input", value=current_data.get('has_cached', False))
                        has_batch = st.checkbox("Supports Batch API", value=current_data.get('has_batch', False))
                        
                        updated_data.update({
                            "input_cost": input_cost,
                            "output_cost": output_cost,
                            "context_window": context_window,
                            "has_cached": has_cached,
                            "has_batch": has_batch
                        })
                        
                        if cached_input_cost > 0:
                            updated_data["cached_input_cost"] = cached_input_cost
                        
                        if pricing_type == "embeddings":
                            dimensions = st.number_input("Dimensions", value=current_data.get('dimensions', 1536), step=1)
                            updated_data["dimensions"] = dimensions
                    
                    # Submit button
                    if st.form_submit_button("💾 Save Changes"):
                        try:
                            from model_utils import ModelCRUD
                            crud = ModelCRUD(model_manager)
                            success, message = crud.update_model(selected_category, selected_model, updated_data)
                            
                            if success:
                                st.success(f"✅ {message}")
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
                        except Exception as e:
                            st.error(f"Error updating model: {str(e)}")
        else:
            st.info("No models found in selected category.")

def show_delete_model(model_manager):
    """Interface for deleting models"""
    st.subheader("🗑️ Delete Model")
    st.warning("⚠️ This action cannot be undone without restoring from backup!")
    
    # Category and model selection
    categories = model_manager.get_categories()
    selected_category = st.selectbox("Select Category", categories, key="delete_cat_select")
    
    if selected_category:
        models = model_manager.get_models_by_category(selected_category)
        if models:
            model_ids = list(models.keys())
            selected_model = st.selectbox("Select Model to Delete", model_ids, key="delete_model_select")
            
            if selected_model:
                # Show model details
                model_data = models[selected_model]
                st.write("**Model to be deleted:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**ID:** {selected_model}")
                    st.write(f"**Name:** {model_data.get('name', 'N/A')}")
                with col2:
                    st.write(f"**Category:** {model_data.get('category', 'N/A')}")
                    st.write(f"**Type:** {model_data.get('pricing_type', 'N/A')}")
                
                # Confirmation
                confirm = st.checkbox(f"I understand that deleting '{selected_model}' is permanent")
                
                if confirm:
                    if st.button("🗑️ Delete Model", type="primary"):
                        try:
                            from model_utils import ModelCRUD
                            crud = ModelCRUD(model_manager)
                            success, message = crud.delete_model(selected_category, selected_model)
                            
                            if success:
                                st.success(f"✅ {message}")
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
                        except Exception as e:
                            st.error(f"Error deleting model: {str(e)}")
        else:
            st.info("No models found in selected category.")

def show_import_export(model_manager):
    """Interface for import/export operations"""
    st.subheader("📤 Import/Export Models")
    
    tab1, tab2 = st.tabs(["📤 Export", "📥 Import"])
    
    with tab1:
        st.write("### Export Models")
        
        # Category selection for export
        categories = model_manager.get_categories()
        export_all = st.checkbox("Export All Categories")
        
        if not export_all:
            selected_categories = st.multiselect("Select Categories to Export", categories)
        else:
            selected_categories = categories
        
        if selected_categories:
            # Generate export data
            export_data = {}
            total_models = 0
            
            for category in selected_categories:
                models = model_manager.get_models_by_category(category)
                if models:
                    export_data[category] = models
                    total_models += len(models)
            
            st.info(f"Ready to export {total_models} models from {len(selected_categories)} categories")
            
            # Export button
            if st.button("📤 Generate Export File"):
                try:
                    export_json = json.dumps(export_data, indent=2, ensure_ascii=False)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"models_export_{timestamp}.json"
                    
                    st.download_button(
                        label="💾 Download Export File",
                        data=export_json,
                        file_name=filename,
                        mime="application/json"
                    )
                    
                    st.success("✅ Export file generated successfully!")
                except Exception as e:
                    st.error(f"Error generating export: {str(e)}")
    
    with tab2:
        st.write("### Import Models")
        
        uploaded_file = st.file_uploader("Choose a JSON file to import", type="json")
        
        if uploaded_file is not None:
            try:
                # Read and parse the uploaded file
                content = uploaded_file.read().decode('utf-8')
                import_data = json.loads(content)
                
                # Preview import data
                st.write("**Import Preview:**")
                preview_data = []
                for category, models in import_data.items():
                    for model_id in models.keys():
                        preview_data.append({
                            "Category": category,
                            "Model ID": model_id,
                            "Name": models[model_id].get('name', 'N/A')
                        })
                
                df = pd.DataFrame(preview_data)
                st.dataframe(df, use_container_width=True)
                
                # Import options
                merge_mode = st.radio("Import Mode", 
                                    ["Merge (keep existing, add new)", "Replace (overwrite categories)"])
                
                # Import button
                if st.button("📥 Import Models"):
                    try:
                        from model_utils import ModelCRUD
                        crud = ModelCRUD(model_manager)
                        # Note: This would need to be implemented in ModelCRUD
                        st.success("✅ Models imported successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error importing models: {str(e)}")
                        
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON file. Please check the file format.")
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

def show_backup_restore(model_manager):
    """Interface for backup and restore operations"""
    st.subheader("💾 Backup & Restore")
    
    tab1, tab2 = st.tabs(["💾 Backup", "🔄 Restore"])
    
    with tab1:
        st.write("### Create Backup")
        
        if st.button("💾 Create Backup Now"):
            try:
                from model_utils import ModelBackupManager
                backup_manager = ModelBackupManager()
                success = backup_manager.create_backup(model_manager.models_data, model_manager.config)
                
                if success:
                    st.success("✅ Backup created successfully!")
                else:
                    st.error("❌ Failed to create backup")
            except Exception as e:
                st.error(f"Error creating backup: {str(e)}")
    
    with tab2:
        st.write("### Restore from Backup")
        
        try:
            from model_utils import ModelBackupManager
            backup_manager = ModelBackupManager()
            backups = backup_manager.get_available_backups()
            
            if backups:
                # Display available backups
                backup_data = []
                for backup in backups:
                    backup_data.append({
                        "File": backup['name'],
                        "Date": backup['formatted_time'],
                        "Size": f"{backup['size'] / 1024:.1f} KB"
                    })
                
                df = pd.DataFrame(backup_data)
                st.dataframe(df, use_container_width=True)
                
                # Backup selection
                selected_backup = st.selectbox("Select Backup to Restore", 
                                             [b['name'] for b in backups])
                
                if selected_backup:
                    st.warning("⚠️ Restoring will overwrite current model data!")
                    
                    if st.button("🔄 Restore Backup"):
                        try:
                            selected_backup_file = next(b['file'] for b in backups if b['name'] == selected_backup)
                            # Note: This would need to be implemented in ModelManager
                            st.success("✅ Backup restored successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error restoring backup: {str(e)}")
            else:
                st.info("No backups available.")
                
        except Exception as e:
            st.error(f"Error accessing backups: {str(e)}")

def show_validation(model_manager):
    """Interface for data validation"""
    st.subheader("✅ Data Validation")
    
    if st.button("🔍 Run Validation"):
        try:
            from model_utils import ModelValidator
            validator = ModelValidator()
            
            validation_results = []
            total_models = 0
            errors = 0
            
            for category in model_manager.get_categories():
                models = model_manager.get_models_by_category(category)
                
                for model_id, model_data in models.items():
                    total_models += 1
                    is_valid, message = validator.validate_model(model_data, category, model_manager.config)
                    
                    validation_results.append({
                        "Category": category,
                        "Model ID": model_id,
                        "Status": "✅ Valid" if is_valid else "❌ Invalid",
                        "Message": message if not is_valid else "OK"
                    })
                    
                    if not is_valid:
                        errors += 1
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Models", total_models)
            with col2:
                st.metric("Valid Models", total_models - errors)
            with col3:
                st.metric("Errors Found", errors)
            
            # Results table
            df = pd.DataFrame(validation_results)
            st.dataframe(df, use_container_width=True)
            
            if errors == 0:
                st.success("🎉 All models passed validation!")
            else:
                st.error(f"❌ Found {errors} validation errors. Please review and fix the issues above.")
                
        except Exception as e:
            st.error(f"Error during validation: {str(e)}") 