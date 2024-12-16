import torch
from torch import nn

class ParameterRegressionModel(nn.Module):
    def __init__(self, 
                 image_height, 
                 image_width, 
                 instrument_param_dim, 
                 num_model_types, 
                 total_param_count,
                 masks,
                 hidden_dims=[512, 256, 128],
                 ):
        """
        Regression model for sparse parameter prediction
        
        Args:
        - image_height (int): Height of input scattering images
        - image_width (int): Width of input scattering images
        - instrument_param_dim (int): Number of instrument parameters
        - num_model_types (int): Number of unique model types
        - total_param_count (int): Total number of unique parameters
        - hidden_dims (list): Hidden layer dimensions
        """
        super(ParameterRegressionModel, self).__init__()
        
        # CNN for image feature extraction
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )
        
        # Determine CNN output size
        with torch.no_grad():
            test_img = torch.zeros(1, 1, image_height, image_width)
            cnn_output_size = self.image_encoder(test_img).shape[1]
        # Embedding for model types
        self.model_type_embed = nn.Embedding(num_model_types, 16)
        
        # Fully connected layers
        fc_input_size = cnn_output_size + instrument_param_dim + 16
        
        fc_layers = []
        prev_dim = fc_input_size
        for dim in hidden_dims:
            fc_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = dim
        
        # Final prediction layer
        self.fc_layers = nn.Sequential(*fc_layers)
        self.param_predictor = nn.Linear(prev_dim, total_param_count)
        self.masks = masks
        
    def forward(self, image, instrument_params, model_type):
        """
        Forward pass of the model
        
        Args:
        - image (torch.Tensor): Input scattering image
        - instrument_params (torch.Tensor): Instrument setup parameters
        - model_type (torch.Tensor): Categorical model type
        
        Returns:
        - torch.Tensor: Predicted sample parameters
        """
        # Extract image features
        image_features = self.image_encoder(image)
        
        # Get model type embeddings
        model_type_embed = self.model_type_embed(model_type)
        
        # Concatenate features
        combined_features = torch.cat([
            image_features, 
            instrument_params, 
            model_type_embed
        ], dim=1)

        # Pass through fully connected layers
        fc_output = self.fc_layers(combined_features)
        
        # Predict parameters
        return self.param_predictor(fc_output) * ~ self.masks[model_type]

