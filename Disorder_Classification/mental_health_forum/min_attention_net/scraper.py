import urllib.request as request
from bs4 import BeautifulSoup
import re
import json
import pandas as pd
import tqdm
from collections import defaultdict

def clean_url_tags(text):
    cleaned = re.sub(re.compile('<.*?>'), '', text)
    cleaned = cleaned.strip() # remove whitespaces
    cleaned = cleaned.replace("\n", "")
    return cleaned

def get_page(page_url):
    #page_url = "https://www.mentalhealthforum.net/forum/thread182350.html?s=c37bf2daabfb1cf270e2a2593c4e81aa"
    req = request.Request(page_url, headers={'User-Agent': "Magic Browser"})
    con = request.urlopen(req)
    page = con.read()
    soup = BeautifulSoup(page)

    contents = soup.find_all("blockquote", class_="postcontent restore ")
    dialog = []
    for content in contents:
        utterances = []
        for paragraph in content.contents:
            cleaned = clean_url_tags(str(paragraph))
            if len(cleaned) > 0:
                utterances.append(cleaned)
        dialog.append("\n".join(utterances))
    return dialog


def composite_url(url, index):
    import re
    url = re.sub("_page", str(index), url)
    return url

def get_threads(vals):
    total_page_count = vals["total_pages"]
    threads = defaultdict(list)
    with tqdm.tqdm(total=total_page_count) as pbar:
        for index in range(1, total_page_count+1):
            if index == 1:
                url = vals["start_url"]
            else:
                url = vals["url"]
            try:
                url = composite_url(url, index)
                req = request.Request(url, headers={'User-Agent' : "Magic Browser"})
                con = request.urlopen(req)
                page = con.read()
                soup = BeautifulSoup(page)

                links = soup.find_all("a", recursive=True)
                for link in links:
                    try:
                        if link.get("class")[0] == "title":
                            threads["url"].append(link.get('href'))
                            threads["title"].append(link.contents[0])
                    except:
                        pass
            except:
                pass

            pbar.update(1)

    return threads


def get_all_threads(type, vals):

    threads = get_threads(vals)
    labels = [type]*len(threads['url'])

    df = pd.DataFrame({**threads, **{"label": labels}})

    df.to_csv(f"./data/{type}threads.csv")

    return df

def get_all_dialogs(df, type):
    print("getting the dialogs")
    df["dialog"] = [""]*len(df)

    with tqdm.tqdm(total=len(list(df.iterrows()))) as pbar:
        for index, row in df.iterrows():
            try:
                dialog = get_page(row["url"])
                df.iloc[index]["dialog"] = dialog
            except:
                print(f"error happened for {row['url']}")

            pbar.update(1)


    df.to_csv(f"./data/{type}_threads_w_dialogs.csv")

    return df


index_links = {
    # "anxiety": {
    #     "total_pages" : 271,
    #     "start_url" : "https://www.mentalhealthforum.net/forum/forum365.html",
    #     "url" : f"https://www.mentalhealthforum.net/forum/mental-health-issues-and-experiences/anxiety-forums/anxiety-forum/index_page.html"
    # },
    # "depression": { "total_pages": 901,
    #     "start_url" : "https://www.mentalhealthforum.net/forum/forum366.html",
    #     "url": f"https://www.mentalhealthforum.net/forum/mental-health-issues-and-experiences/depression-forums/depression-forum/index_page.html",
    # },
    # "PTSD": { "total_pages": 39,
    #     "start_url" : "https://www.mentalhealthforum.net/forum/forum41.html",
    #     "url": f"https://www.mentalhealthforum.net/forum/mental-health-issues-and-experiences/anxiety-forums/post-traumatic-stress-disorder-forum/index_page.html"
    #     },
    "Bipolar": {
        "total_pages": 837,
        "start_url": "https://www.mentalhealthforum.net/forum/forum37.html",
        "url": f"https://www.mentalhealthforum.net/forum/mental-health-issues-and-experiences/bipolar-disorder-cyclothymia-and-manic-depression-forum/index_page.html"
    },
    "self-harm": {
        "total_pages": 174,
        "start_url": "https://www.mentalhealthforum.net/forum/forum29.html",
        "url": f"https://www.mentalhealthforum.net/forum/mental-health-issues-and-experiences/self-harm-forum/index_page.html"
    },
    "Schizophrenia": {
        "total_pages" : 345,
        "start_url" :"https://www.mentalhealthforum.net/forum/forum31.html",
        "url" : f"https://www.mentalhealthforum.net/forum/mental-health-issues-and-experiences/schizophrenia-forum/index_page.html"
    },
    "Hearing Voices" : {
        "total_pages" : 142,
        "start_url" : "https://www.mentalhealthforum.net/forum/forum36.html",
        "url" : f"https://www.mentalhealthforum.net/forum/mental-health-issues-and-experiences/hearing-voices/hearing-voices-forum/index_page.html"
    }

}


def main():
    for type, vals in index_links.items():
        threads_df = get_all_threads(type, vals)
        print(f"got {len(threads_df)} threads with {len(threads_df['url'].unique())} uniques")
        df = get_all_dialogs(threads_df, type)



if __name__=="__main__":
    main()



