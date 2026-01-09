# Class 15-16: Advanced Recommender Systems Topics
> See Class 13 for Matrix Factorization core content.  
> This note covers advanced topics.

## Advanced Techniques

### Deep Learning for Recommendations

```python
import tensorflow as tf

# Neural Collaborative Filtering
def build_ncf(n_users, n_items, embedding_dim=50):
    user_input = tf.keras.Input(shape=(1,))
    item_input = tf.keras.Input(shape=(1,))
    
    user_embed = tf.keras.layers.Embedding(n_users, embedding_dim)(user_input)
    item_embed = tf.keras.layers.Embedding(n_items, embedding_dim)(item_input)
    
    user_vec = tf.keras.layers.Flatten()(user_embed)
    item_vec = tf.keras.layers.Flatten()(item_embed)
    
    concat = tf.keras.layers.Concatenate()([user_vec, item_vec])
    dense1 = tf.keras.layers.Dense(128, activation='relu')(concat)
    dense2 = tf.keras.layers.Dense(64, activation='relu')(dense1)
    output = tf.keras.layers.Dense(1)(dense2)
    
    model = tf.keras.Model([user_input, item_input], output)
    model.compile(optimizer='adam', loss='mse')
    return model
```

### Context-Aware Recommendations

```python
# Include contextual features (time, location, device)
def context_aware_prediction(user, item, context_features):
    base_score = mf_model.predict(user, item)
    context_adjustment = context_model.predict(context_features)
    return base_score * context_adjustment
```

**Note:** Refer to **Class 13** for:
- Complete matrix factorization theory
- SVD, ALS algorithms
- Full implementation examples
- Comprehensive exam preparation
