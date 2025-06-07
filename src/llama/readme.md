# LLM Neapykantos Kalbos Aptikimo Modelis

Naudojamas Llama 2 modelis, skirtas neapykantos ir įžeidžios kalbos aptikimui lietuviškuose tekstuose. Modelis klasifikuoja tekstus į tris kategorijas: `neapykanta`, `įžeidus`, `neutralus`.

## Reikalavimai

- Python 3.x
- `llama_cpp` biblioteka
- `pandas` biblioteka

## Diegimas

1. Įdiekite reikiamas priklausomybes:
    ```bash
    pip install llama-cpp-python pandas
    ```

2. Naudokit duomenų CSV failą, kurio struktūra turi būti tokia:
    ```csv
    data,labels
    "Teksto pavyzdys", "etiketė"
    ```

3. Atsisiųskite llama gguf modelį
    https://www.jottacloud.com/s/1580a97507a3efe4c14b0d3cede3caf3584/list/

## Naudojimas

Paleiskite pagrindinį skriptą su šiais argumentais:

```bash
python main.py --model_path ./Lt-Llama-13b-q8.gguf --csv_path ./data.csv --output_path ./rezultatai.txt
```