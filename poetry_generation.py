import difflib

def generate_poems(sentences, rhymes):
    poems = []
    matcher1 = difflib.SequenceMatcher()
    matcher2 = difflib.SequenceMatcher()
    for i in range(len(sentences)):
        matcher1.set_seq1(sentences[i])
        for j in range(len(sentences)):
            if i >= j:
                continue
            if sentences[i] == sentences[j]:
                continue
            if sentences[i].split()[-1].lower() == sentences[j].split()[-1].lower():
                continue
            matcher1.set_seq2(sentences[j])
            ratio1 = matcher1.ratio()
            if ratio1 >= 0.7:
                continue
            for k1 in range(len(rhymes[i])):
                matcher2.set_seq1(rhymes[i][k1])
                for k2 in range(len(rhymes[j])):
                    if rhymes[i][k1] == rhymes[j][k2]:
                        continue
                    if rhymes[i][k1].split()[-1].lower() == rhymes[j][k2].split()[-1].lower():
                        continue
                    matcher2.set_seq1(rhymes[j][k2])
                    ratio2 = matcher2.ratio()
                    if ratio2 >= 0.7:
                        continue
                    poems.append(
                        (
                            "\n".join([sentences[i], sentences[j], rhymes[i][k1], rhymes[j][k2]]),
                            max(ratio1, ratio2)
                        )
                    )
    return [it2[0] for it2 in sorted(poems, key=lambda it: it[1])]


if __name__ == "__main__":
    sentences = ['Греф остался президентом Сбербанка',
                 'Герман Греф: признаков выздоровления экономики нет',
                 'Греф конкретизировал обещания Владимира Путина',
                 'Саакашвили отстранил собственного пресс-секретаря за\xa0антисемитизм',
                 'Греф готовит для России общественное телевидение',
                 'Ющенко отказался ехать на\xa0Петербургский экономический форум',
                 'Бывший вице-премьер Сирии арестован по\xa0обвинению в\xa0коррупции',
                 'Алексей Кудрин заменит заморозку цен их\xa0стабилизацией',
                 'Кудрин прокомментировал свою отставку',
                 'Путин ликвидировал Министерство по\xa0делам национальностей']

    rhymes = ['Греф остался президентом Сбербанка',
              'Герман Греф: признаков выздоровления экономики нет',
              'Греф конкретизировал обещания Владимира Путина',
              'Саакашвили отстранил собственного пресс-секретаря за\xa0антисемитизм',
              'Греф готовит для России общественное телевидение',
              'Ющенко отказался ехать на\xa0Петербургский экономический форум',
              'Бывший вице-премьер Сирии арестован по\xa0обвинению в\xa0коррупции',
              'Алексей Кудрин заменит заморозку цен их\xa0стабилизацией',
              'Кудрин прокомментировал свою отставку',
              'Путин ликвидировал Министерство по\xa0делам национальностей']

    for poetry in generate_poems(sentences, rhymes):
        print(poetry)
        print()
