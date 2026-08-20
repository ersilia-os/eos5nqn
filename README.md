# GNEProp Escherichia coli antibiotic activity

Assesses antibacterial activity against Escherichia coli, including a permeability-compromised strain alongside the wild type so that intrinsic potency can be separated from failure to penetrate. Scalia and colleagues at Genentech coupled a high-throughput phenotypic screen with deep learning applied at ultra-large scale, searching well beyond the screened set and surfacing scaffolds structurally unlike known antibacterials. Predictions rest on growth inhibition and carry no information about mechanism of action.

This model was incorporated on 2025-12-10.Last packaged on 2026-05-20.

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
- **Output Dimension:** `2`
- **Output Consistency:** `Fixed`
- **Interpretation:** Probability of Escherichia coli growth inhibition in wild-type and permeability-compromised strains.

Below are the **Output Columns** of the model:
| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| tolc_activity | float | high | Probability score of inhibiting the Escherichia coli tolC strain using a subset of the HTS screen |
| hts_activity | float | high | Probability score of inhibiting the Escherichia coli tolC strain using a large HTS screen |


### Source and Deployment
- **Source:** `Local`
- **Source Type:** `External`
- **DockerHub**: [https://hub.docker.com/r/ersiliaos/eos5nqn](https://hub.docker.com/r/ersiliaos/eos5nqn)
- **Docker Architecture:** `AMD64`
- **S3 Storage**: [https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos5nqn.zip](https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos5nqn.zip)

### Resource Consumption
- **Model Size (Mb):** `2599`
- **Environment Size (Mb):** `2469`
- **Image Size (Mb):** `4478.84`

**Computational Performance (seconds):**
- 10 inputs: `39.55`
- 100 inputs: `73.95`
- 10000 inputs: `-1`

### References
- **Source Code**: [https://github.com/Genentech/gneprop](https://github.com/Genentech/gneprop)
- **Publication**: [https://doi.org/10.1101/2024.09.11.612340](https://doi.org/10.1101/2024.09.11.612340)
- **Publication Type:** `Preprint`
- **Publication Year:** `2024`
- **Ersilia Contributor:** [miquelduranfrigola](https://github.com/miquelduranfrigola)

### License
This package is licensed under a [GPL-3.0](https://github.com/ersilia-os/ersilia/blob/master/LICENSE) license. The model contained within this package is licensed under a [Apache-2.0](LICENSE) license.

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
