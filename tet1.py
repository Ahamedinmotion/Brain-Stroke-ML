from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the model
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Changed to 1 unit with sigmoid activation
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # This works for binary classification
              metrics=['accuracy'])

# Setup ImageDataGenerators for data augmentation and rescaling
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    r"C:\Users\khais\OneDrive\Desktop\brainstroke\dataset\Train",  # Path to training data
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'  # Binary classification
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    r"C:\Users\khais\OneDrive\Desktop\brainstroke\dataset\Validation",  # Path to validation data
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'  # Binary classification
)

# Fit the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32
)

model.save('brainstroke_model.h5')
