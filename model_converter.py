import torch
import torch.onnx as onnx
import torch.nn as nn
import torchvision.models as models

#Function to Convert to ONNX
def convert_to_onnx(model : nn.Module, input_dims : tuple, output_path : str): # str or nn.Module
	# Let's create a dummy input tensor
	dummy_input = torch.rand(1, *input_dims, requires_grad=True)

	# set the model to inference mode
	model.eval()

	input_names = "Input"
	output_names = "Output"
	dynamic_axis_name = "batch_size"

	# Export the model
	onnx.export(model,										# model being run
		dummy_input,										# model input (or a tuple for multiple inputs)
		output_path,										# where to save the model
		export_params=True,									# store the trained parameter weights inside the model file
		opset_version=10,									# the ONNX version to export the model to
		do_constant_folding=True,							# whether to execute constant folding for optimization
		input_names = [input_names],						# the model's input names
		output_names = [output_names],						# the model's output names
		dynamic_axes={input_names : {0 : dynamic_axis_name},		# variable length axes
						output_names : {0 : dynamic_axis_name}})
	print(" ")
	print('Model has been converted to ONNX')

if __name__ == "__main__":
	# prepare the model, loading the base model structure and substituting the last layer
	model = models.vgg16(pretrained=False)

	last_layer = model.classifier[-1]
	classification_layer = nn.Linear(in_features=last_layer.in_features, out_features=133, bias=True)
	model.classifier[-1] = classification_layer

	# load the saved model parameters
	path = "model_transfer.pt"
	state_dict = torch.load(path)
	model.load_state_dict(state_dict)

	# Freeze all layers
	for parameter in model.parameters():
		parameter.requires_grad = False

	# Conversion to ONNX
	convert_to_onnx(model, (3, 224, 224), "model_transfer.onnx")
