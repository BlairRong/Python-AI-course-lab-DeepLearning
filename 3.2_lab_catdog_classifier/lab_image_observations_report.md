# Lab Report: Cat vs Dog Classification

## Data Inspection Observations

After inspecting the dataset, I observed several factors that make this classification task challenging despite having only two classes:

1. **Resolution Differences分辨率差异**: Images vary significantly in size - some are high-resolution close-ups while others are low-resolution or distant shots.

2. **Background Complexity背景复杂性**: Backgrounds range from simple solid colors to complex outdoor scenes with vegetation, furniture, or other objects that can confuse the model.

3. **Lighting Conditions光照条件**: Images have varying lighting - some are well-lit studio shots, others are dark indoor photos or have harsh outdoor lighting creating shadows.

4. **Breed and Pose Variation品种和姿势差异**: Dogs and cats come in vastly different shapes, sizes, and colors. Animals appear in different poses - sitting, standing, running, sleeping - making it hard to learn consistent features.

5. **Partial Occlusion部分遮挡**: Some images show only parts of animals (face only, body partially hidden) rather than complete animals.

These variations mean the model must learn to recognize the essential features of cats and dogs while ignoring irrelevant variations in background, lighting, and pose.