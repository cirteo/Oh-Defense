class AttentionMasker:
    """Attention head masker for analyzing safe attention heads in language models
    
    This class provides a mechanism to temporarily modify (mask) specific attention
    head weights in language models to analyze their impact on model safety outputs.
    
    Attributes:
        model: The language model to be analyzed
        head_mask: Configuration of heads to mask {(layer, head): [qkv types list]}
        mask_type: Type of masking, either "scale_mask" or "zero_mask"
        scale_factor: The scaling factor when using "scale_mask"
        hooks: List of model hooks
        original_weights: Dictionary to store original weights
        num_heads: Number of attention heads per layer
        head_dim: Dimension of each attention head
    """
    def __init__(self, model, head_mask=None, mask_type="scale_mask", scale_factor=1e-5):
        """Initialize the attention head masker
        
        Args:
            model: The language model to be analyzed
            head_mask: Configuration of heads to mask {(layer, head): [qkv types list]}
            mask_type: Type of masking, either "scale_mask" or "zero_mask"
            scale_factor: The scaling factor when using "scale_mask"
        """
        self.model = model
        self.head_mask = head_mask
        self.mask_type = mask_type
        self.scale_factor = scale_factor
        self.hooks = []
        self.original_weights = {}
        
        # Get parameters from model configuration
        self.num_heads = getattr(model.config, "num_attention_heads", 32)
        self.head_dim = getattr(model.config, "hidden_size", 4096) // self.num_heads
    
    def apply_hooks(self):
        """Apply hooks to the attention layers of the language model
        
        This method adds pre-forward and post-forward hooks to each attention layer
        of the model to temporarily modify and restore attention head weights during
        forward propagation.
        """
        if self.head_mask is None:
            return
            
        self.remove_hooks()
        
        # Add hooks for language model
        for layer_idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'self_attn'):
                pre_hook = layer.self_attn.register_forward_pre_hook(
                    lambda module, inputs, layer_idx=layer_idx: 
                        self._pre_attention_hook(module, inputs, layer_idx)
                )
                self.hooks.append(pre_hook)
                
                post_hook = layer.self_attn.register_forward_hook(
                    lambda module, inputs, output, layer_idx=layer_idx:
                        self._post_attention_hook(module, inputs, output, layer_idx)
                )
                self.hooks.append(post_hook)

    def _pre_attention_hook(self, module, inputs, layer_idx):
        """Modify attention weights before forward propagation
        
        Modifies weights of specific attention heads according to head_mask
        configuration before model forward pass begins.
        
        Args:
            module: The current module being processed
            inputs: The module inputs
            layer_idx: The current layer index
        """
        if not hasattr(module, 'q_proj'):
            return
            
        # Process heads that need to be masked
        for (masked_layer, masked_head), qkv_types in self.head_mask.items():
            if masked_layer == layer_idx:
                key = f"layer_{layer_idx}_{masked_head}"
                
                # Process q/k/v projections
                for proj_type in qkv_types:
                    proj_name = f"{proj_type}_proj"
                    if hasattr(module, proj_name):
                        proj = getattr(module, proj_name)
                        start_idx = masked_head * self.head_dim
                        end_idx = (masked_head + 1) * self.head_dim
                        
                        # Save original weights
                        self.original_weights[f"{key}_{proj_type}"] = proj.weight[start_idx:end_idx].clone()
                        # Apply mask
                        if self.mask_type == "scale_mask":
                            proj.weight.data[start_idx:end_idx] *= self.scale_factor
                        elif self.mask_type == "zero_mask":
                            proj.weight.data[start_idx:end_idx] = 0
    
    def _post_attention_hook(self, module, inputs, output, layer_idx):
        """Restore attention weights after forward propagation
        
        Restores the modified attention head weights after module's forward
        propagation completes.
        
        Args:
            module: The current module being processed
            inputs: The module inputs
            output: The module outputs
            layer_idx: The current layer index
            
        Returns:
            The module outputs
        """
        if not hasattr(module, 'q_proj'):
            return output
            
        # Restore weights
        for key, weight in list(self.original_weights.items()):
            if key.startswith(f"layer_{layer_idx}_"):
                parts = key.split('_')
                masked_head = int(parts[2])
                proj_type = parts[3]
                
                start_idx = masked_head * self.head_dim
                end_idx = (masked_head + 1) * self.head_dim
                
                proj_name = f"{proj_type}_proj"
                if hasattr(module, proj_name):
                    proj = getattr(module, proj_name)
                    proj.weight.data[start_idx:end_idx] = weight
                    del self.original_weights[key]
        
        return output
    
    def remove_hooks(self):
        """Remove all hooks and restore original weights
        
        Removes all hooks added to the model and ensures all modified weights
        are restored to their original state.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Ensure all original weights are restored
        for key, weight in list(self.original_weights.items()):
            parts = key.split('_')
            layer_idx, head_idx, proj_type = int(parts[1]), int(parts[2]), parts[3]
            
            try:
                module = self.model.model.layers[layer_idx].self_attn
                start_idx = head_idx * self.head_dim
                end_idx = (head_idx + 1) * self.head_dim
                
                if proj_type == 'q' and hasattr(module, 'q_proj'):
                    module.q_proj.weight.data[start_idx:end_idx] = weight
                elif proj_type == 'k' and hasattr(module, 'k_proj'):
                    module.k_proj.weight.data[start_idx:end_idx] = weight
                elif proj_type == 'v' and hasattr(module, 'v_proj'):
                    module.v_proj.weight.data[start_idx:end_idx] = weight
                
            except Exception as e:
                print(f"Error when restoring weights: {key}, Error: {str(e)}")
                
        self.original_weights = {}


class Qwen2AttentionMasker:
    def __init__(self, model, head_mask=None, mask_type="scale_mask", scale_factor=1e-5):
        self.model = model
        self.head_mask = head_mask
        self.mask_type = mask_type
        self.scale_factor = scale_factor
        self.hooks = []
        self.original_weights = {}
        self.original_biases = {}
        
        # Get parameters from model configuration
        self.num_heads = getattr(model.config, "num_attention_heads", 32)
        self.head_dim = getattr(model.config, "hidden_size", 4096) // self.num_heads
    
    def apply_hooks(self):
        """Apply hooks to the attention layers of the language model.
        
        This method adds pre-forward and post-forward hooks to each attention layer
        of the model to temporarily modify and restore attention head weights during
        forward propagation.
        """
        if self.head_mask is None:
            return
            
        self.remove_hooks()
        
        # Add hooks for the language model
        for layer_idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'self_attn'):
                pre_hook = layer.self_attn.register_forward_pre_hook(
                    lambda module, inputs, layer_idx=layer_idx: 
                        self._pre_attention_hook(module, inputs, layer_idx)
                )
                self.hooks.append(pre_hook)
                
                post_hook = layer.self_attn.register_forward_hook(
                    lambda module, inputs, output, layer_idx=layer_idx:
                        self._post_attention_hook(module, inputs, output, layer_idx)
                )
                self.hooks.append(post_hook)

    def _pre_attention_hook(self, module, inputs, layer_idx):
        if not hasattr(module, 'q_proj'):
            return
            
        # Process heads that need to be masked
        for (masked_layer, masked_head), qkv_types in self.head_mask.items():
            if masked_layer == layer_idx:
                key = f"layer_{layer_idx}_{masked_head}"
                
                # Process q/k/v projections
                for proj_type in qkv_types:
                    proj_name = f"{proj_type}_proj"
                    if hasattr(module, proj_name):
                        proj = getattr(module, proj_name)
                        start_idx = masked_head * self.head_dim
                        end_idx = (masked_head + 1) * self.head_dim
                        
                        # Save original weights
                        self.original_weights[f"{key}_{proj_type}"] = proj.weight[start_idx:end_idx].clone()
                        
                        # Apply weight mask
                        if self.mask_type == "scale_mask":
                            proj.weight.data[start_idx:end_idx] *= self.scale_factor
                        elif self.mask_type == "zero_mask":
                            proj.weight.data[start_idx:end_idx] = 0
                        
                        # If there is a bias, process the bias as well
                        if hasattr(proj, 'bias') and proj.bias is not None:
                            # Save original bias
                            self.original_biases[f"{key}_{proj_type}"] = proj.bias[start_idx:end_idx].clone()
                            
                            # Apply bias mask
                            if self.mask_type == "scale_mask":
                                proj.bias.data[start_idx:end_idx] *= self.scale_factor
                            elif self.mask_type == "zero_mask":
                                proj.bias.data[start_idx:end_idx] = 0
    
    def _post_attention_hook(self, module, inputs, output, layer_idx):
        """Restore attention weights and biases after forward propagation.
        
        After the module's forward propagation is complete, restore the modified
        attention head weights and biases.
        
        Args:
            module: The current module being processed.
            inputs: The module's inputs.
            output: The module's outputs.
            layer_idx: The index of the current layer.
            
        Returns:
            The module's output.
        """
        if not hasattr(module, 'q_proj'):
            return output
            
        # Restore weights
        for key, weight in list(self.original_weights.items()):
            if key.startswith(f"layer_{layer_idx}_"):
                parts = key.split('_')
                masked_head = int(parts[2])
                proj_type = parts[3]
                
                start_idx = masked_head * self.head_dim
                end_idx = (masked_head + 1) * self.head_dim
                
                proj_name = f"{proj_type}_proj"
                if hasattr(module, proj_name):
                    proj = getattr(module, proj_name)
                    proj.weight.data[start_idx:end_idx] = weight
                    del self.original_weights[key]
        
        # Restore biases
        for key, bias in list(self.original_biases.items()):
            if key.startswith(f"layer_{layer_idx}_"):
                parts = key.split('_')
                masked_head = int(parts[2])
                proj_type = parts[3]
                
                start_idx = masked_head * self.head_dim
                end_idx = (masked_head + 1) * self.head_dim
                
                proj_name = f"{proj_type}_proj"
                if hasattr(module, proj_name) and hasattr(getattr(module, proj_name), 'bias') and getattr(module, proj_name).bias is not None:
                    proj = getattr(module, proj_name)
                    proj.bias.data[start_idx:end_idx] = bias
                    del self.original_biases[key]
        
        return output
    
    def remove_hooks(self):
        """Remove all hooks and restore original weights and biases.
        
        Remove all hooks added to the model and ensure that all modified weights
        and biases are restored to their original state.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Ensure all original weights are restored
        for key, weight in list(self.original_weights.items()):
            parts = key.split('_')
            layer_idx, head_idx, proj_type = int(parts[1]), int(parts[2]), parts[3]
            
            try:
                module = self.model.model.layers[layer_idx].self_attn
                start_idx = head_idx * self.head_dim
                end_idx = (head_idx + 1) * self.head_dim
                
                if proj_type == 'q' and hasattr(module, 'q_proj'):
                    module.q_proj.weight.data[start_idx:end_idx] = weight
                elif proj_type == 'k' and hasattr(module, 'k_proj'):
                    module.k_proj.weight.data[start_idx:end_idx] = weight
                elif proj_type == 'v' and hasattr(module, 'v_proj'):
                    module.v_proj.weight.data[start_idx:end_idx] = weight
            except Exception as e:
                print(f"Error restoring weight: {key}, Error: {str(e)}")
                
        self.original_weights = {}
        
        # Ensure all original biases are restored
        for key, bias in list(self.original_biases.items()):
            parts = key.split('_')
            layer_idx, head_idx, proj_type = int(parts[1]), int(parts[2]), parts[3]
            
            try:
                module = self.model.model.layers[layer_idx].self_attn
                start_idx = head_idx * self.head_dim
                end_idx = (head_idx + 1) * self.head_dim
                
                if proj_type == 'q' and hasattr(module, 'q_proj') and hasattr(module.q_proj, 'bias') and module.q_proj.bias is not None:
                    module.q_proj.bias.data[start_idx:end_idx] = bias
                elif proj_type == 'k' and hasattr(module, 'k_proj') and hasattr(module.k_proj, 'bias') and module.k_proj.bias is not None:
                    module.k_proj.bias.data[start_idx:end_idx] = bias
                elif proj_type == 'v' and hasattr(module, 'v_proj') and hasattr(module.v_proj, 'bias') and module.v_proj.bias is not None:
                    module.v_proj.bias.data[start_idx:end_idx] = bias
            except Exception as e:
                print(f"Error restoring bias: {key}, Error: {str(e)}")
                
        self.original_biases = {}


class PhiAttentionMasker:
    def __init__(self, model, head_mask=None, mask_type="scale_mask", scale_factor=1e-5):
        self.model = model
        self.head_mask = head_mask
        self.mask_type = mask_type
        self.scale_factor = scale_factor
        self.hooks = []
        self.original_weights = {}
        
        # Get parameters from model configuration
        self.num_heads = getattr(model.config, "num_attention_heads", 32)
        self.head_dim = getattr(model.config, "hidden_size", 4096) // self.num_heads
    
    def apply_hooks(self):
        """Apply hooks to the attention layers of the language model.
        
        This method adds pre-forward and post-forward hooks to each attention layer
        of the model to temporarily modify and restore attention head weights during
        forward propagation.
        """
        if self.head_mask is None:
            return
            
        self.remove_hooks()
        
        # Add hooks for the language model
        for layer_idx, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'self_attn'):
                pre_hook = layer.self_attn.register_forward_pre_hook(
                    lambda module, inputs, layer_idx=layer_idx: 
                        self._pre_attention_hook(module, inputs, layer_idx)
                )
                self.hooks.append(pre_hook)
                
                post_hook = layer.self_attn.register_forward_hook(
                    lambda module, inputs, output, layer_idx=layer_idx:
                        self._post_attention_hook(module, inputs, output, layer_idx)
                )
                self.hooks.append(post_hook)

    def _pre_attention_hook(self, module, inputs, layer_idx):
        """Modify attention weights before forward propagation.
        
        Before the model's forward propagation begins, modify the weights of
        specific attention heads according to the head_mask configuration.
        
        Args:
            module: The current module being processed.
            inputs: The module's inputs.
            layer_idx: The index of the current layer.
        """
        if not hasattr(module, 'qkv_proj'):
            return
            
        # Process heads that need to be masked
        for (masked_layer, masked_head), qkv_types in self.head_mask.items():
            if masked_layer == layer_idx:
                key = f"layer_{layer_idx}_{masked_head}"
                
                # Get qkv_proj
                qkv_proj = module.qkv_proj
                
                # Calculate the offset of q, k, v parts in the merged weight
                hidden_size = self.num_heads * self.head_dim
                
                # Assume the shape of qkv_proj weight is [3*hidden_size, hidden_size]
                # where the first hidden_size rows are for q, the next for k, and then v
                
                for proj_type_idx, proj_type in enumerate(qkv_types):
                    # Determine the position of q/k/v in qkv_proj
                    if proj_type == 'q':
                        offset = 0
                    elif proj_type == 'k':
                        offset = hidden_size
                    elif proj_type == 'v':
                        offset = 2 * hidden_size
                    else:
                        continue
                    
                    # Calculate the index range for the specific head in qkv_proj
                    start_idx = offset + masked_head * self.head_dim
                    end_idx = offset + (masked_head + 1) * self.head_dim
                    
                    # Save original weights
                    self.original_weights[f"{key}_{proj_type}"] = qkv_proj.weight[start_idx:end_idx].clone()
                    
                    # Apply mask
                    if self.mask_type == "scale_mask":
                        qkv_proj.weight.data[start_idx:end_idx] *= self.scale_factor
                    elif self.mask_type == "zero_mask":
                        qkv_proj.weight.data[start_idx:end_idx] = 0
    
    def _post_attention_hook(self, module, inputs, output, layer_idx):
        """Restore attention weights after forward propagation.
        
        After the module's forward propagation is complete, restore the modified
        attention head weights.
        
        Args:
            module: The current module being processed.
            inputs: The module's inputs.
            output: The module's outputs.
            layer_idx: The index of the current layer.
            
        Returns:
            The module's output.
        """
        if not hasattr(module, 'qkv_proj'):
            return output
            
        # Restore weights
        for key, weight in list(self.original_weights.items()):
            if key.startswith(f"layer_{layer_idx}_"):
                parts = key.split('_')
                masked_head = int(parts[2])
                proj_type = parts[3]
                
                # Get qkv_proj
                qkv_proj = module.qkv_proj
                
                # Calculate the offset of q, k, v parts in the merged weight
                hidden_size = self.num_heads * self.head_dim
                
                if proj_type == 'q':
                    offset = 0
                elif proj_type == 'k':
                    offset = hidden_size
                elif proj_type == 'v':
                    offset = 2 * hidden_size
                else:
                    continue
                
                start_idx = offset + masked_head * self.head_dim
                end_idx = offset + (masked_head + 1) * self.head_dim
                
                # Restore original weights
                qkv_proj.weight.data[start_idx:end_idx] = weight
                del self.original_weights[key]
        
        return output
    
    def remove_hooks(self):
        """Remove all hooks and restore original weights.
        
        Remove all hooks added to the model and ensure that all modified weights
        are restored to their original state.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Ensure all original weights are restored
        for key, weight in list(self.original_weights.items()):
            parts = key.split('_')
            layer_idx, head_idx, proj_type = int(parts[1]), int(parts[2]), parts[3]
            
            try:
                module = self.model.layers[layer_idx].self_attn
                
                if hasattr(module, 'qkv_proj'):
                    hidden_size = self.num_heads * self.head_dim
                    
                    if proj_type == 'q':
                        offset = 0
                    elif proj_type == 'k':
                        offset = hidden_size
                    elif proj_type == 'v':
                        offset = 2 * hidden_size
                    else:
                        continue
                    
                    start_idx = offset + head_idx * self.head_dim
                    end_idx = offset + (head_idx + 1) * self.head_dim
                    
                    module.qkv_proj.weight.data[start_idx:end_idx] = weight
            except Exception as e:
                print(f"Error restoring weight: {key}, Error: {str(e)}")
                
        self.original_weights = {}
