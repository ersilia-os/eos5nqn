# GNEProp Escherichia coli antibiotic activity

The GNEProp model was trained based on a high-throughput screening effort against a sensitized Escherichia coli strain. The model was able to identify new antibiotic candidates in a billion-scale screening effort. small-molecule high-throughput screening with a deep-learning-based virtual screening approach to uncover new antibacterial compounds. Robustness of the model was validated with respect to out-of-distribution generalization and activity cliff prediction.

This model was incorporated on 2025-12-10.


## Information
### Identifiers
- **Ersilia Identifier:** `eos5nqn`
- **Slug:** `gneprop-ecoli`

### Domain
- **Task:** `Annotation`
- **Subtask:** `Activity prediction`
- **Biomedical Area:** `Antimicrobial resistance`
- **Target Organism:** `Escherichia coli`
- **Tags:** `Antimicrobial activity`

### Input
- **Input:** `Compound`
- **Input Dimension:** `1`

### Output
- **Output Dimension:** `1`
- **Output Consistency:** `Fixed`
- **Interpretation:** Probability of inhibiting a sensitized strain of Escherichia coli.

Below are the **Output Columns** of the model:
| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| tolc_activity | float | high | Probability score of inhibiting the Escherichia coli tolC strain |


### Source and Deployment
- **Source:** `Local`
- **Source Type:** `External`
- **S3 Storage**: [https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos5nqn.zip](https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos5nqn.zip)

### Resource Consumption
- **Model Size (Mb):** `2576`
- **Environment Size (Mb):** `8426`


### References
- **Source Code**: [https://github.com/Genentech/gneprop](https://github.com/Genentech/gneprop)
- **Publication**: [https://www.nature.com/articles/s41587-025-02814-6](https://www.nature.com/articles/s41587-025-02814-6)
- **Publication Type:** `Peer reviewed`
- **Publication Year:** `2025`
- **Ersilia Contributor:** [miquelduranfrigola](https://github.com/miquelduranfrigola)

### License
This package is licensed under a [GPL-3.0](https://github.com/ersilia-os/ersilia/blob/master/LICENSE) license. The model contained within this package is licensed under a [GPL-3.0-or-later](LICENSE) license.

**Notice**: Ersilia grants access to models _as is_, directly from the original authors, please refer to the original code repository and/or publication if you use the model in your research.


## Use
To use this model locally, you need to have the [Ersilia CLI](https://github.com/ersilia-os/ersilia) installed.
The model can be **fetched** using the following command:
```bash
# fetch model from the Ersilia Model Hub
ersilia fetch eos5nqn
```
Then, you can **serve**, **run** and **close** the model as follows:
```bash
# serve the model
ersilia serve eos5nqn
# generate an example file
ersilia example -n 3 -f my_input.csv
# run the model
ersilia run -i my_input.csv -o my_output.csv
# close the model
ersilia close
```

## About Ersilia
The [Ersilia Open Source Initiative](https://ersilia.io) is a tech non-profit organization fueling sustainable research in the Global South.
Please [cite](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff) the Ersilia Model Hub if you've found this model to be useful. Always [let us know](https://github.com/ersilia-os/ersilia/issues) if you experience any issues while trying to run it.
If you want to contribute to our mission, consider [donating](https://www.ersilia.io/donate) to Ersilia!
