import logging
from typing import Dict
import os
import openai
import pandas as pd
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

client = OpenAI(
    api_key= os.getenv("OPENAI_API_KEY")
)


class HateSpeechDetector:
    def __init__(self, model):
        self.model = model
        self.system_prompt = """
        Tu esi neapykantos kalbos aptikimo ekspertas. Tavo užduotis - įvertinti tekstą dėl neapykantos ar įžeidžios kalbos.
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

            # response = client.chat.completions.create(
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": f"Komentaras, kurį reikia įvertinti: {text} \n\n"
                                   f"Atsakyme parašyk tik vieną žodį: neapykanta, įžeidus arba neutralus.",
                    },
                ],
                temperature=0.1,
                max_tokens=50,
            )
            print(response)
            result = response.choices[0].message.content.strip().lower()
            logging.info(f"Input: {text[:100]}... | Classification: {result}")
            return {"text": text, "classification": result}

        except Exception as e:
            logging.error(f"Error processing text: {str(e)}")
            return {"text": text, "classification": "error"}


def runner(csv_path: str, model: str = "gpt-4o"):
    detector = HateSpeechDetector(model)

    # read csv
    df = pd.read_csv(csv_path)
    comments = df["data"].tolist()
    real_labels = df["labels"].tolist()

    with open(f"./results-{model}.csv", "w") as f:
        for i, text in enumerate(comments):
            result = detector.detect(text)
            f.write(f"{text[:50]}\t{real_labels[i]}\t{result['classification']}\n")


if __name__ == "__main__":
    import argparse
    # before running this script, make sure to set your OpenAI API key as an environment variable:
    # export OPENAI_API_KEY=your_api_key_here

    parser = argparse.ArgumentParser(description="Run hate speech detection on a CSV file.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use")
    args = parser.parse_args()

    runner(csv_path=args.csv_path, model=args.model)
