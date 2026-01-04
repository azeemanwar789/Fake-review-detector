from utils import FraudReviewAnalyzer

# Load model
analyzer = FraudReviewAnalyzer()

# Test different reviews
test_reviews = [
    "FREE MONEY!!! Click here now!!!",
    "This is a normal product review.",
    "The product works okay. Not bad.",
    "OMG BEST PRODUCT EVER!!! BUY NOW!!!",
    "I don't recommend this product.",
]

print("\n" + "=" * 60)
print("🧪 TESTING MODEL WITH DIFFERENT REVIEWS")
print("=" * 60)

for i, review in enumerate(test_reviews):
    print(f"\n📝 Test {i+1}: '{review}'")
    result = analyzer.analyze_review(review)
    
    if result['success']:
        print(f"   Result: {'SPAM' if result['is_fraud'] else 'SAFE'}")
        print(f"   Probability: {result['fraud_probability']:.4f}")
        print(f"   Raw Probability: {result.get('raw_probability', 0):.4f}")
    else:
        print(f"   ERROR: {result.get('error', 'Unknown')}")

print("\n" + "=" * 60)
print("✅ TEST COMPLETE")
print("=" * 60)