import json
from typing import Any, \
	Type, \
	Callable, \
	Dict, \
	get_args, \
	get_origin, \
	List, \
	Optional
import inspect


def pythonTypeToJsonSchema(pythonType: Type[Any]) -> str:
	"""
	Converts Python type annotations to JSON schema types.

	:param pythonType: A Python type or type annotation.
	:return: A string representing the JSON schema type.
	"""
	# Define a mapping of Python types to JSON schema types
	typeMapping: dict[Type[Any], str] = {
		bool      : 'boolean',
		int       : 'integer',
		float     : 'number',
		str       : 'string',
		type(None): 'null'  # Handle the NoneType for optional type annotations
	}

	# If the type is directly in the mapping, return it
	if pythonType in typeMapping:
		return typeMapping[pythonType]

	# Handle generic types from typing module
	originType: Optional[Type[Any]] = get_origin(pythonType)
	if originType:
		# Example for List, Dict; you can extend this as needed
		if issubclass(originType,
					  list):
			return 'array'
		elif issubclass(originType,
						dict):
			return 'object'

	# If a custom mapping for a user-defined class is needed
	if hasattr(pythonType,
			   '__name__'):
		customTypeMapping: dict[str, str] = {
			# 'CustomClass': 'object',  # Example of a custom class to JSON type
			# You can add more custom mappings here
		}
		return customTypeMapping.get(pythonType.__name__,
									 'string')

	# If the type is a typing.Union or similar, you might want to handle it differently
	# For example, Optional[int] could be translated as type: ['integer', 'null']

	# Fallback to a string type if not specified or unknown
	return 'string'


def parseDocstring(docstring: str) -> Dict[str, Any]:
	"""
	Parses the docstring to find parameter descriptions and additional metadata.

	:param docstring: The docstring from which to extract parameter descriptions and metadata.
	:return: A dictionary with parameter names as keys and their descriptions and metadata as values.
	"""
	lines = docstring.split('\n')
	paramDescriptions: Dict[str, Dict[str, Any]] = {}
	currentParam: str = ""
	currentKey: str = ""
	parsingReturn: bool = False  # Added a flag for return description parsing

	for line in lines:
		line = line.strip()
		if line.startswith(':param'):
			_, param_name = line.split(' ',
									   1)
			currentParam, _ = param_name.split(':',
											   1)
			currentParam = currentParam.strip()
			paramDescriptions[currentParam] = {
				'description': ''}
			currentKey = 'description'
		elif line.startswith(':return:'):
			# Handle return description
			returnDescription: str = line.replace(':return:',
												  '').strip()
			currentParam = ""  # Reset current parameter when parsing return description
			parsingReturn = True
		elif line.startswith('-'):
			# This line contains additional metadata for the parameter
			if currentParam:
				metaParts = line.lstrip('- ').split(': ',
													1)
				if len(metaParts) == 2:
					key, value = metaParts
					key = key.strip()
					value = value.strip()
					# Cast the value to the appropriate type
					if key in ['minimum',
							   'maximum',
							   'default']:
						try:
							value = float(value) if '.' in value else int(value)
						except ValueError:
							pass  # If casting fails, retain the string value
					if key in ['validInputs',
							   'invalidInputs']:
						try:
							validationList: list[str] = value.split(",")
						except ValueError:
							pass
					paramDescriptions[currentParam][key] = value
					currentKey = key
		elif currentParam and currentKey:
			# Continue the current parameter's description or metadata
			paramDescriptions[currentParam][currentKey] += ' ' + line.strip()

	# Strip any extra whitespace from the descriptions
	for param, desc in paramDescriptions.items():
		desc['description'] = desc['description'].strip()

	return paramDescriptions


def generateParamProperties(name: str,
							itemType: any,
							param: inspect.Parameter,
							paramDescriptions: Dict[str, Any],
							paramModifiers: Dict[str, Any],
							inMatryoshka: bool = False) -> Dict[str, Any]:
	paramDescription = paramDescriptions.get(name,
											 {}).get('description',
													 '')
	jsonType: str = pythonTypeToJsonSchema(itemType)

	if jsonType != "array":
		outProperties = {
			"type": jsonType}

		if not inMatryoshka:
			outProperties['description'] = paramDescription

		for mod, value in paramModifiers.items():
			outProperties[mod] = value
		return outProperties

	else:
		arrayItemType = get_args(itemType)[0] if get_args(itemType) else 'Any'
		outProperties = {
			"type": jsonType}
		if not inMatryoshka:
			outProperties['description'] = paramDescription
		outProperties['items'] = generateParamProperties(name=name,
														 itemType=arrayItemType,
														 param=param,
														 paramDescriptions=paramDescriptions,
														 paramModifiers=paramModifiers,
														 inMatryoshka=True)
		return outProperties


def generateToolsConfig(function: Callable[..., Any]) -> str:
	"""
	Generates the 'tools' section of the OpenAI API configuration for a given function.

	:param function: A callable function to generate the tools configuration for.
	:return: A JSON string representing the tools configuration.
	"""
	signature: inspect.Signature = inspect.signature(function)
	docstring: str = inspect.getdoc(function) or "No description provided."
	functionName: str = function.__name__
	paramDescriptions: Dict[str, Any] = parseDocstring(docstring)

	print(paramDescriptions)
	# Prepare the list for 'required' fields
	requiredList: List[str] = [name for name, param in signature.parameters.items() if
							   param.default is inspect.Parameter.empty]

	properties: Dict[str, Dict[str, Any]] = {}
	for name, param in signature.parameters.items():
		paramType: Any = param.annotation if param.annotation is not inspect.Parameter.empty else 'Any'

		modifierProperties: Dict[str, Any] = {}

		validModifiers: list[str] = ['minimum',
									 'maximum',
									 'validInputs',
									 'invalidInputs']

		for modifier in validModifiers:
			if modifier in paramDescriptions.get(name,
												 {}):
				modifierProperties[modifier] = paramDescriptions[name][modifier]

		properties[name] = generateParamProperties(name=name,
												   itemType=paramType,
												   param=param,
												   paramModifiers=modifierProperties,
												   paramDescriptions=paramDescriptions)

	tool: Dict[str, Any] = {
		"type"    : "function",
		"function": {
			"name"       : functionName,
			"description": docstring,
			"parameters" : {
				"type"      : "object",
				"properties": properties,
				"required"  : requiredList
			}
		}
	}

	return json.dumps(tool,
					  indent=4)


def exampleFunction(decision: bool,
					confidence: list[list[float]],
					answer: str,
					reasoning: str = "") -> str:
	'''
	Forces the AI to make a yes/no choice on the situation at hand.

	:param decision:
		The answer to the decision, true for yes, false for no.
	:param confidence:
		Floating point confidence values of the decisions.
		- minimum: 0.0
		- maximum: 100.0
	:param answer:
		Yes or no string which represents the decision.
		- validInputs: YES, NO
	:param reasoning:
		The justification as to why the decision was made.

	:return: A string explaining the decision.
	'''
	return "Decision made."


"""
[{
    "type": "function",
    "function": {
        "name": "makeDecision",
        "description": "Forces the AI to make a yes no choice on the situation at hand.",
        "parameters": {
            "type": "object",
            "properties": {
                      "decision": {"type": "boolean",
                             "description": "The answer to the decision, true for yes, false for no."},
                "confidence": {"type": "number",
                               "minimum": 0,
                               "maximum": 1,
                               "description": "Floating point confidence value of the decision, 1 for totally confident, 0 for not confident at all, and all percentage values inbetween"},
                "reasoning": {
                    "type": "string",
                    "description": "The justification as to why the decision was made."}
                },
            "required": [
                "decision"
                ]
            }
        }
    }]
"""
