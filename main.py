import streamlit as st
import numpy as np
import pickle
import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

df = pd.read_csv("divorce.csv")


# Load the trained SGD model and scaler
@st.cache_resource
def load_model_and_scaler():
    with open("sgd_model.pkl", "rb") as model_file, open("scaler.pkl", "rb") as scaler_file:
        sgd_model = pickle.load(model_file)
        scaler = pickle.load(scaler_file)
    return sgd_model, scaler

sgd_model, scaler = load_model_and_scaler()

# Define the questions
questions = [
    "If one of us apologizes when our discussion deteriorates, the discussion ends.",
    "I know we can ignore our differences, even if things get hard sometimes.",
    "When we need it, we can take our discussions with my spouse from the beginning and correct it.",
    "When I discuss with my spouse, to contact him will eventually work.",
    "The time I spent with my wife is special for us.",
    "We don't have time at home as partners.",
    "We are like two strangers who share the same environment at home rather than family.",
    "I enjoy our holidays with my wife.",
    "I enjoy traveling with my wife.",
    "Most of our goals are common to my spouse.",
    "I think that one day in the future, when I look back, I see that my spouse and I have been in harmony with each other.",
    "My spouse and I have similar values in terms of personal freedom.",
    "My spouse and I have similar sense of entertainment.",
    "Most of our goals for people (children, friends, etc.) are the same.",
    "Our dreams with my spouse are similar and harmonious.",
    "We're compatible with my spouse about what love should be.",
    "We share the same views about being happy in our life with my spouse.",
    "My spouse and I have similar ideas about how marriage should be.",
    "My spouse and I have similar ideas about how roles should be in marriage.",
    "My spouse and I have similar values in trust.",
    "I know exactly what my wife likes.",
    "I know how my spouse wants to be taken care of when she/he is sick.",
    "I know my spouse's favorite food.",
    "I can tell you what kind of stress my spouse is facing in her/his life.",
    "I have knowledge of my spouse's inner world.",
    "I know my spouse's basic anxieties.",
    "I know what my spouse's current sources of stress are.",
    "I know my spouse's hopes and wishes.",
    "I know my spouse very well.",
    "I know my spouse's friends and their social relationships.",
    "I feel aggressive when I argue with my spouse.",
    "When discussing with my spouse, I usually use expressions such as ‘you always’ or ‘you never’.",
    "I can use negative statements about my spouse's personality during our discussions.",
    "I can use offensive expressions during our discussions.",
    "I can insult my spouse during our discussions.",
    "I can be humiliating when we have discussions.",
    "My discussion with my spouse is not calm.",
    "I hate my spouse's way of opening a subject.",
    "Our discussions often occur suddenly.",
    "We're just starting a discussion before I know what's going on.",
    "When I talk to my spouse about something, my calm suddenly breaks.",
    "When I argue with my spouse, I only go out and I don't say a word.",
    "I mostly stay silent to calm the environment a little bit.",
    "Sometimes I think it's good for me to leave home for a while.",
    "I'd rather stay silent than discuss with my spouse.",
    "Even if I'm right in the discussion, I stay silent to hurt my spouse.",
    "When I discuss with my spouse, I stay silent because I am afraid of not being able to control my anger.",
    "I feel right in our discussions.",
    "I have nothing to do with what I've been accused of.",
    "I'm not actually the one who's guilty about what I'm accused of.",
    "I'm not the one who's wrong about problems at home.",
    "I wouldn't hesitate to tell my spouse about her/his inadequacy.",
    "When I discuss, I remind my spouse of her/his inadequacy.",
    "I'm not afraid to tell my spouse about her/his incompetence."
]

# Streamlit app title
st.title("SGD Classifier: Divorce Prediction ")

# Navigation bar using selectbox
page = st.selectbox("Choose a page", ["Prediction", "Visualization","Comparison of ML Algorithms"])

# Prediction Page
if page == "Prediction":
    # User instructions
    st.markdown("Answer the following questions on a scale from 0 (Strongly Disagree) to 4 (Strongly Agree).")

    # Collect user inputs with numbering using Markdown and separate sliders
    responses = []
    for i, question in enumerate(questions, start=1):
        st.markdown(f"**{i}. {question}**")  # Display question number and text in bold
        response = st.slider(
            label="",  # Empty label since the question is displayed above
            min_value=0,
            max_value=4,
            value=2,
            step=1,
            key=f"question_{i}"
        )
        responses.append(response)

    # Predict button
    if st.button("Predict"):
        # Convert responses to numpy array and preprocess
        user_responses = np.array(responses).reshape(1, -1)
        user_responses_scaled = scaler.transform(user_responses)

        # Make prediction
        prediction = sgd_model.predict(user_responses_scaled)


        # Display the result with probability
        if prediction[0] == 1:

            st.error(f"**Prediction:** Likely to Divorce ")
        else:

            st.success(f"**Prediction:** Not Likely to Divorce ")



# Visualization Page
elif page == "Visualization":
    # Select the features and target variable
    X = df.drop("Divorce_Y_N", axis=1)
    y = df["Divorce_Y_N"]
    # Calculate the correlation matrix
    correlation_matrix = df.corr()


    target_correlations = correlation_matrix['Divorce_Y_N'].drop('Divorce_Y_N')


    sorted_correlations = target_correlations.abs().sort_values(ascending=True)


    features = sorted_correlations.head(6).index

    # Streamlit title
    st.title("Correlation Matrix of Selected Features")


    # Optionally, visualize the correlations using a heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[features.to_list()].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    plt.title('Correlation Matrix of Selected Features')
    st.pyplot(fig)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA to reduce the dimensionality to 2 components
    pca = PCA(n_components=0.95)  # Retain 95% of the variance
    X_pca = pca.fit_transform(X_scaled)

    # Streamlit Visualization Page
    st.title("PCA Visualization")


    # 2D Scatter Plot
    st.subheader("PCA: 2D Projection")
    fig_2d = plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA: 2D Projection")
    st.pyplot(fig_2d)

    # 3D Scatter Plot (only if the PCA has 3 components)
    if X_pca.shape[1] >= 3:
        st.subheader("PCA: 3D Projection")
        fig_3d = plt.figure(figsize=(8, 6))
        ax = fig_3d.add_subplot(111, projection='3d')
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', edgecolor='k')
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("PCA: 3D Projection")
        st.pyplot(fig_3d)
    else:
        st.warning("The data does not have enough dimensions for a 3D PCA projection.")
    # Calculate the correlation
    st.subheader("Correlation with Divorce_Y_N")
    corr = df.corr()['Divorce_Y_N'].drop('Divorce_Y_N')
    sort_corr = corr.abs().sort_values(ascending=False)[:20]
    st.write("Top 20 features with highest absolute correlation with 'Divorce_Y_N':")
    st.write(sort_corr)

    # Show the bar plot for correlation
    st.subheader("Correlation Bar Plot")
    fig_corr = plt.figure(figsize=(9, 8))
    sns.barplot(x=sort_corr.index, y=sort_corr)
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Absolute Correlation')
    plt.title('Top 20 Features Correlated with Divorce_Y_N')
    st.pyplot(fig_corr)

    # Add advice summary based on the top features
    st.subheader("Relationship Advice to Avoid Divorce")

    # Advice based on the top 20 features
    advice = """
    Here are some valuable pieces of advice based on the features most strongly correlated with divorce:

    1. **Good Communication**: Regular, open, and empathetic conversations with your spouse help in resolving conflicts and understanding each other's needs.
    2. **Shared Values**: Aligning on values related to personal freedom, trust, and mutual respect can strengthen your bond.
    3. **Quality Time**: Spending time together on holidays and during travels creates shared experiences and strengthens relationships.
    4. **Emotional Support**: Being able to understand each other's stresses, hopes, and anxieties fosters emotional intimacy.
    5. **Conflict Resolution**: Learning how to resolve conflicts without escalating them and using "I" statements instead of accusatory language can prevent emotional harm.
    6. **Appreciation**: Show appreciation for the time spent together and the little things your partner does.
    7. **Balance Time at Home**: Prioritize spending quality time at home, making your relationship a priority amidst the busy schedules.
    8. **Trust and Transparency**: Be open and transparent about your thoughts, feelings, and expectations.
    9. **Mutual Understanding**: Take the time to learn about your spouse’s needs, likes, and dislikes.
    10. **Healthy Boundaries**: Ensure that you both have healthy emotional boundaries while being supportive of each other.
    11. **Problem-Solving Together**: Facing challenges together and solving problems as a team strengthens the partnership.
    12. **Respect and Care**: Showing respect and care during disagreements, even when you disagree, keeps the relationship harmonious.
    13. **Emotional Connection**: Stay emotionally connected and avoid using harmful language during arguments.
    14. **Be Willing to Apologize**: Apologizing and admitting when you are wrong helps in mending rifts and restoring harmony.
    15. **Self-Awareness**: Being aware of your own emotional needs and how they impact the relationship can make you a better partner.
    16. **Kindness and Affection**: Regularly showing affection and kindness helps maintain a loving atmosphere.
    17. **Seek Professional Help**: If communication breaks down, seeking help from a therapist can provide tools to rebuild the relationship.
    18. **Focus on the Positive**: Acknowledge and celebrate each other’s achievements and qualities.
    19. **Understanding Each Other’s Love Languages**: Understanding how your spouse expresses and receives love can improve your connection.
    20. **Work Together Toward Common Goals**: Build shared goals and dreams for the future to keep both partners aligned and motivated.
    """

    st.write(advice)




    image1 = Image.open("images.jpeg")


    st.image(image1, caption="Happy Marriage", use_container_width=True)
elif page == "Comparison of ML Algorithms":

    st.title("Comparison of Metrics for Each Algorithms")

    
    st.subheader("Table of Metrics by Classifiers")
    image1 = Image.open("table.png")
    st.image(image1, caption="Table of Metrics by Classifiers", use_column_width=True)

    st.subheader("Precision")
    image1 = Image.open("Precision.png")
    st.image(image1, caption="Precision", use_container_width=True)

    st.subheader("Accuracy")
    image1 = Image.open("accuracy.png")
    st.image(image1, caption="Accuracy", use_container_width=True)

    st.subheader("Recall")
    image1 = Image.open("Recall.png")
    st.image(image1, caption="Recall", use_container_width=True)


    st.subheader("Specificity")
    image1 = Image.open("Spe.png")
    st.image(image1, caption="Specificity", use_container_width=True)

    st.subheader("F1-Score")
    image1 = Image.open("f1score.png")
    st.image(image1, caption="F1-Score", use_container_width=True)

    st.subheader("ROC AUC Score")
    image1 = Image.open("RocAuc.png")
    st.image(image1, caption="ROC AUC Score", use_container_width=True)

    st.subheader("ROC and AUC")
    image1 = Image.open("Roc.png")
    st.image(image1, caption="ROC and Auc", use_container_width=True)
    st.markdown("""
    ### Based on the provided metrics, the SGD Classifier stands out with:

    - The highest accuracy (0.98).
    - Near-perfect precision and recall for both classes, leading to a balanced and robust model.

    Why Choose SGD Classifier?
    - It achieves the best overall performance while maintaining high precision and recall for both classes.
    - It is lightweight and computationally efficient, making it suitable for large datasets or real-time predictions.
    """)
else:
        st.warning("Please make a prediction first on the Prediction page.")
