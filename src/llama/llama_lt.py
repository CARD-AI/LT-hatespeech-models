import logging
from typing import Dict
import pandas as pd
from llama_cpp import Llama
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class HateSpeechDetector:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.llm = Llama(
            model_path=model_path, chat_format="llama-2", device=device, n_ctx=2048
        )

        self.system_prompt = """Tu esi neapykantos kalbos aptikimo ekspertas. Tavo užduotis - įvertinti tekstą dėl neapykantos ar įžeidžios kalbos.

        Atsakyme pateik TIK VIENĄ iš šių žodžių (be papildomo teksto):
        - neapykanta: tekstas, kuris skatina neapykantą, smurtą ar diskriminaciją dėl:
          * rasės, tautybės, etninės kilmės
          * religijos ar tikėjimo
          * lyties, seksualinės orientacijos
          * negalios
          * amžiaus
        Pavyzdžiai neapykantos kalbos:
          * Gaidys gaidys, ministras gaidys. Toki gurkli kad tureciau, visus jankius pamyleciau
          * o smarvė-baisu,meilė putleristanui ir jo carui-okupantui dvokia
          * na ir tupas tamsta, ar tupa......skaityt ir nesuprast kas nužudė
          * zydu bijo, kaip velnias kryziaus

        - įžeidus: tekstas, kuris yra:
          * įžeidžiantis, bet neskatina neapykantos
          * naudoja necenzūrinę kalbą
          * žemina ar tyčiojasi iš kitų
          * skleidžia dezinformaciją ar šmeižtą
        Pavyzdžiai įžeidžios kalbos:
          * Vaikai ir anukai rusu kalbos nemoka, tai skirta, matyt, senjoritoms ir ju širdies draugams?
          * is kur cia ta nurasyta pupu dede vel istrauke
          * Demokratiški idiotu puseje nedalyvauja.
          * VISU GALU MEISTRE IR BEBRU KARALIENE NEGALI NEPEZETI

        - neutralus: tekstas be neapykantos ar įžeidžių elementų
        Pavyzdžiai neutralaus turinio:
          * Tai neimanoma
          * tik nuo kada krokodilas su zuikiu susitaike
          * Cia geruoju nesibaigs
        """

    def detect(self, text: str) -> Dict:
        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": f"Komentaras, kurį reikia įvertinti: {text} \n\n"
                        f"Atsakyme parašyk tik vieną žodį: neapykanta, įžeidus arba neutralus.",
                    },
                ],
                temperature=0.1,
                repeat_penalty=1.1,
                max_tokens=50,
            )

            result = response["choices"][0]["message"]["content"].strip().lower()
            logging.info(f"Input: {text[:100]}... | Classification: {result}")
            return {"text": text, "classification": result}

        except Exception as e:
            logging.error(f"Error processing text: {str(e)}")
            return {"text": text, "classification": "error"}


def runner(model_path: str, csv_path: str, output_path: str):
    detector = HateSpeechDetector(model_path=model_path)

    # read csv
    df = pd.read_csv(csv_path)
    comments = df["data"].tolist()
    real_labels = df["labels"].tolist()

    with open(output_path, "w") as f:
        for i, text in enumerate(comments):
            result = detector.detect(text)
            f.write(f"{text[:50]}\t{real_labels[i]}\t{result['classification']}\n")


if __name__ == "__main__":
    # Argument parser to get model path and CSV path from command line
    parser = argparse.ArgumentParser(description="Run hate speech detection model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the Llama model"
    )
    parser.add_argument(
        "--csv_path", type=str, required=True, help="Path to the input CSV file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./rezultatai.txt",
        help="Path to save the output results",
    )

    args = parser.parse_args()

    runner(
        model_path=args.model_path, csv_path=args.csv_path, output_path=args.output_path
    )
