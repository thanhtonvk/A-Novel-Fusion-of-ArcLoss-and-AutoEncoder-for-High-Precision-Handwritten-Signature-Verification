# A Novel Fusion of ArcLoss and AutoEncoder for High-Precision Handwritten Signature Verification

This project introduces a novel hybrid approach for handwritten signature verification that combines ArcLoss and AutoEncoder techniques with state-of-the-art deep learning backbone networks, namely ResNet50 and MobileNet. The proposed method aims to enhance the discriminative power and robustness of signature verification models, effectively addressing the challenges posed by signature variability and skilled forgeries.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Usage](#usage)
6. [Contributing](#contributing)
7. [License](#license)

## Introduction

Handwritten signature verification plays a crucial role in ensuring the security and authenticity of financial transactions, legal documents, and personal identification. However, accurate signature verification remains challenging due to the inherent variability in individual signing styles and the potential for skilled forgeries. This project proposes a novel approach that integrates ArcLoss and AutoEncoder techniques with deep learning backbone networks to improve the performance of signature verification models.

## Dataset

The project utilizes a comprehensive signature dataset consisting of genuine and forged signatures. The dataset is divided into two parts:

- **Training set**: Contains 549 classes, with a total of 50,546 signature images.
- **Testing set**: Includes 13,353 signature pairs, with 6,734 genuine pairs and 6,619 forged pairs.

The dataset folder in this repository contains the necessary files and instructions for accessing and using the signature dataset.

## Methodology

The proposed approach combines ArcLoss and AutoEncoder techniques with state-of-the-art deep learning backbone networks. The key components of the methodology include:

- **Preprocessing**: Signature region extraction and background whitening using color thresholding in the HSV color space.
- **Feature Extraction**: Utilizing ResNet50 and MobileNet as backbone networks for extracting discriminative features from signature images.
- **ArcLoss**: Integrating ArcLoss to enhance the discriminative power of the learned features by enforcing a large angular margin between classes.
- **AutoEncoder**: Employing AutoEncoders to learn compact and noise-robust representations of signature images, capturing intrinsic patterns and variations.

The source code folder in this repository contains the implementation of the proposed methodology, along with detailed documentation and instructions for running the experiments.

## Results

The proposed approach achieves state-of-the-art performance on the handwritten signature verification task. On the private test set, the method obtains the following results:

- **ResNet50**: Accuracy of 99.7%, FAR of 0.01, and FRR of 0.02
- **MobileNet**: Accuracy of 99.3%, FAR of 0.09, and FRR of 0.03

These results demonstrate the effectiveness of the proposed hybrid approach in accurately distinguishing between genuine and forged signatures.

## Usage

To use this project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/handwritten-signature-verification.git
2. Install the required dependencies as specified in the requirements.txt file.
3. [Request signature](https://forms.gle/NMvgMJrQ1TYmfbF89) 
3. Download the signature dataset and place it in the datasets/images folder.
4. Extract mask from signature
   ```bash 
   python extract_mask.py
> [!WARNING]  
> Please check the label again after extracting the mak before training.
5. Training 
   ```bash
   python train.py configs/mbf
4. Follow the instructions provided in the source code folder to preprocess the data, train the models, and evaluate their performance.
5. Refer to the documentation and code comments for detailed explanations of each component of the project.

## Contributing
Contributions to this project are welcome. If you encounter any issues, have suggestions for improvements, or want to add new features, please open an issue or submit a pull request. Make sure to follow the contribution guidelines outlined in the repository.

## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute the code for academic or commercial purposes.

For any further questions or inquiries, please contact the project maintainer at thanhtonvk@gmail.com.
