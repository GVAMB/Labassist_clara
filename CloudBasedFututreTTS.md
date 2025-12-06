# Azure Text-to-Speech (TTS) Integration

This project integrates Azure Cognitive Services' Text-to-Speech (TTS) API to convert text into natural-sounding speech. Azure TTS uses advanced neural voices that can be customized in terms of voice selection, speech rate, pitch, and volume.

## Prerequisites

Before you can use Azure TTS, you'll need:

- **Azure Cognitive Services Account**: You need an Azure account with the **Speech API** enabled. [Create an Azure Speech Service](https://azure.microsoft.com/en-us/services/cognitive-services/speech-services/).
- **API Key and Region**: After creating the Speech API resource, obtain the **API Key** and **Region** from the Azure portal. You'll use these to authenticate your application with Azure TTS.

### Step-by-Step Setup

### 1. Create Azure Cognitive Services Account

- Go to the [Azure portal](https://portal.azure.com/).
- Navigate to **Create a resource** → **AI + Machine Learning** → **Speech**.
- Choose **Speech API** and set up your **Speech Service** resource.
- Once your service is created, go to the **Keys and Endpoint** section and grab your **API Key** and **Region**.

### 2. Install Required Dependencies

You'll need to install the official Azure SDK for Python to interact with Azure's Speech API. Follow these steps to install it:

1. **Install the Azure Speech SDK**:

   ```bash
   pip install azure-cognitiveservices-speech
