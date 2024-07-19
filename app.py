from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
import numpy as np
import matplotlib.pyplot as plt
import io
from models import Generator, Discriminator
import tensorflow as tf

app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing; adjust as needed for production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Load pre-trained models
generator = Generator(model_path='generator_model.h5')
discriminator = Discriminator(model_path='discriminator_model.h5')

class ImagesRequest(BaseModel):
    images: list[list[list[float]]]

@app.get("/generate")
async def generate_image(num_images: int = Query(1, gt=0)):
    """
    Generates images using the pre-trained generator model and returns them as a PNG file.

    Query Parameters:
        - num_images (int): The number of images to generate. Default is 1.

    Returns:
        - A PNG file of generated images.
    """
    try:
        # Generate images
        images = generator.generate(num_images).numpy()

        # Create an image grid
        fig, axs = plt.subplots(1, num_images, figsize=(num_images * 2, 2), squeeze=False)
        for i in range(num_images):
            axs[0, i].imshow(images[i].reshape(28, 28), cmap='gray')
            axs[0, i].axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close()

        return StreamingResponse(buf, media_type="image/png")
    
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/discriminator")
async def evaluate_discriminator(request: ImagesRequest):
    """
    Evaluates the discriminator on provided images and returns the prediction results.

    Request Body:
        - images (list of lists): A list of image arrays (28x28) to evaluate.

    Returns:
        - JSON response with the discriminator's predictions.
    """
    try:
        images = np.array(request.images)
        if images.shape[1:] != (28, 28):
            raise ValueError("Images must be 28x28 arrays.")

        # Flatten images and make predictions
        images = images.reshape(-1, 28 * 28).astype(np.float32)
        predictions = discriminator.model.predict(images)

        return {"predictions": predictions.tolist()}
    
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
