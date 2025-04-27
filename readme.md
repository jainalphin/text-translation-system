# Report

## 1. Tokenizer Analysis and Integration Strategy

**Tokenizers Analyzed:**
- mBART-50
- IndicBERT
- SUTRA

**Evaluation Metrics:**
- Vocabulary size comparison
- Token-per-word ratio (efficiency)
- Token length distribution
- Character-per-token efficiency
- Word-level inefficiency detection

**Final Choice:**  
After a detailed comparison, the **mBART-50 tokenizer** was selected for integration.  
It offered the best balance between English and Hindi tokenization efficiency, while also ensuring compatibility with the chosen model architecture and minimizing token inflation across languages.

---

## 2. Model Selection and Training Methodology

**Models Explored:**
1. `facebook/mbart-large-50-many-to-many-mmt`
2. mBART

**Key Observations:**
- The original mBART model is relatively outdated, and its fine-tuning documentation was poorly maintained.
- The **facebook/mbart-large-50-many-to-many-mmt** model is widely adopted for multilingual translation tasks, with better community support and updated resources.
- Based on ease of access, fine-tuning stability, and proven success in translation tasks, **facebook/mbart-large-50-many-to-many-mmt** was selected.

---

## 3. Evaluation Approach

**Metrics Used:**
- **BLEU Score**: Measures n-gram overlap between generated and reference translations.
- **chrF Score**: Measures character-level precision and recall, useful for morphologically rich languages like Hindi.

Both metrics were calculated after model training to quantitatively assess translation quality.  
Sample translations were also reviewed manually for qualitative assessment.

---

## 4. Key Takeaways and Suggested Next Steps

**Key Takeaways:**
- Pretrained multilingual models like mBART-50 adapt effectively even with small-scale fine-tuning.
- Tokenizer efficiency has a direct impact on training cost and sequence length.
- Dataset size and quality are critical drivers of translation performance.

**Next Steps:**
- **Expand the Dataset**: Currently, only ~1000 records were used. Increasing data volume will significantly enhance performance.
- **Improve Document Alignment**: Investigate better Hindi-English document alignment techniques, such as bilingual sentence alignment algorithms or semi-automated cleaning.

---

# Submission Summary

- Tokenizer evaluation and final selection completed
- Model fine-tuning and translation pipeline developed
- Evaluation reports generated using BLEU and chrF metrics
- Recommendations for scaling and improvement provided
