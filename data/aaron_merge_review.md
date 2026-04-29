# YOE Knowledge Graph — Aaron's Merge Review

Below are entity-merge candidates that automated heuristics couldn't decide. For each cluster, the **canonical** name is what the merged entity will be called. Members listed under it would be merged INTO the canonical IF approved.

For each member, we provide the LLM-generated description from the KG plus a transcript citation showing where the name was actually mentioned in a podcast. Mark each one **YES** (merge into canonical) or **NO** (separate entity).

---

## `c0000` — Canonical: **Y on Earth Community** (ORGANIZATION)

_Canonical KG description:_ An initiative focused on sustainability and environmental education, where individuals gather to learn about and promote biochar and other technologies.

- **Member:** `Why on Earth` (mc=19, eps=[17, 47, 56, 58, 67])
  - _Description:_ An initiative or platform focused on sustainability and environmental education, often involving discussions and interviews on related topics.
  - _Heuristic:_ via=fuzzy score=0.944 overlap=0.31
  - _Source:_ [ep 17 at 20:51] "You know, one of the things Adam that I discovered doing research when I was writing Why on Earth is that our fossil energy prices generally are bouncing along a band of price ranges that behave like a company."
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `Y-Earth community` (mc=4, eps=[66, 100, 102, 129])
  - _Description:_ A community focused on environmental justice issues, advocating for awareness and action against pollution.
  - _Heuristic:_ user-flagged for Aaron citation review (phonetic noise)
  - _Source:_ [ep 66 at 04:20] "And we know through many of our other ambassadors and friends and allies at the Y-Earth community,"
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `Y-Earth Community` (mc=4, eps=[52, 61, 85, 154])
  - _Description:_ A community focused on stewardship and sustainability, hosting a podcast series to promote positive environmental actions.
  - _Heuristic:_ user-flagged for Aaron citation review (phonetic noise)
  - _Source:_ [ep 52 at 05:18] "Well, we love that because our Y-Earth community is all about action."
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `Y Honors Community` (mc=3, eps=[68, 111, 147])
  - _Description:_ An organization focused on sustainability and community engagement, hosting a podcast series.
  - _Heuristic:_ user-flagged for Aaron citation review (phonetic noise)
  - _Source:_ [ep 68 at 00:07] "Welcome to the Y Honors Community Podcast. Today I'm here with Artim Nicholkov and we are"
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `Y Honors community` (mc=3, eps=[85, 92, 131])
  - _Description:_ A community focused on recognizing and celebrating achievements in sustainability and environmental stewardship.
  - _Heuristic:_ user-flagged for Aaron citation review (phonetic noise)
  - _Source:_ [ep 85 at 10:17] "Honors community is a series of achievement badges that we can award to folks in our ambassador"
  - _(member name not verbatim; matched token `honors`)_
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `White Honors Community` (mc=2, eps=[54, 92])
  - _Description:_ A community focused on promoting sustainability and personal well-being through various practices and themes.
  - _Heuristic:_ user-flagged for Aaron citation review (phonetic noise)
  - _Source:_ [ep 54 at 01:51] "And Katie is also on the board of the Y Honors community."
  - _(member name not verbatim; matched token `honors`)_
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `Why Honors Community` (mc=2, eps=[115, 158])
  - _Description:_ A community focused on sharing knowledge and education, involved in various projects including a podcast.
  - _Heuristic:_ user-flagged for Aaron citation review (phonetic noise)
  - _Source:_ [ep 115 at 00:10] "Welcome to the Why On Earth Community Podcast. Today we have a really special episode for you."
  - _(member name not in transcript; likely Whisper drift — found canonical `Y on Earth Community` instead)_
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `Wieners community` (mc=2, eps=[68, 120])
  - _Description:_ A community focused on sustainability and regenerative practices, inviting engagement and participation.
  - _Heuristic:_ user-flagged for Aaron citation review (phonetic noise)
  - _Source:_ [ep 68 at 56:17] "get your products and some of the proceeds come back to support the Y on Earth community"
  - _(member name not in transcript; likely Whisper drift — found canonical `Y on Earth Community` instead)_
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `Winers community` (mc=2, eps=[76, 98])
  - _Description:_ A community focused on facilitating substantial rapid change in sustainable practices globally.
  - _Heuristic:_ user-flagged for Aaron citation review (phonetic noise)
  - _Source:_ [ep 76] _(name not in transcript verbatim; likely LLM extraction artifact)_  Episode: "Episode 76 – Charles Orgbon III, Deloitte, Environmental Liability & Sustainability Consultant"
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `YNRF community` (mc=2, eps=[168, 169])
  - _Description:_ A community involved in sustainability efforts and international partnerships.
  - _Heuristic:_ user-flagged for Aaron citation review (phonetic noise)
  - _Source:_ [ep 168 at 00:11] "Welcome to the Y on Earth Community Podcast. I'm your host, Aaron William Perry. And today we're visiting with Ken LaRoe, the founder and chairman of Climate First Bank. Hi Ken, how you doing?"
  - _(member name not in transcript; likely Whisper drift — found canonical `Y on Earth Community` instead)_
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `Why On Earth` (mc=2, eps=[103, 170])
  - _Description:_ A platform focused on sustainability and environmental education, providing resources and support for regenerative practices.
  - _Heuristic:_ via=fuzzy score=0.944 overlap=0.23
  - _Source:_ [ep 103 at 33:50] "give a shout out to all of the sponsors making this podcast series possible, uh, through our Why On Earth"
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `Wired Earth Community Network` (mc=1, eps=[112])
  - _Description:_ A community network focused on addressing various global issues and promoting sustainability.
  - _Heuristic:_ user-flagged for Aaron citation review (phonetic noise)
  - _Source:_ [ep 112 at 43:13] "offer discounts to folks in our whyonnereth community audience and network. And you can find links"
  - _(member name not verbatim; matched token `network`)_
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `Wine and Earth Community` (mc=1, eps=[58])
  - _Description:_ A community focused on mobilizing efforts for environmental sustainability and regenerative practices.
  - _Heuristic:_ user-flagged for Aaron citation review (phonetic noise)
  - _Source:_ [ep 58 at 40:37] "Well, let me just say here and to the why on Earth community, I, community is what we're"
  - _(member name not in transcript; likely Whisper drift — found canonical `Y on Earth Community` instead)_
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `YonEarth Communities` (mc=1, eps=[46])
  - _Description:_ An organization focused on stewardship and sustainability, hosting a podcast series to promote environmental awareness.
  - _Heuristic:_ via=fuzzy+semantic score=0.000 overlap=0.23
  - _Source:_ [ep 46 at 29:42] "So let me just remind our audience, this is the Why on Earth Communities Stewardship and"
  - _(member name not verbatim; matched token `communities`)_
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `Y-Energ community` (mc=1, eps=[78])
  - _Description:_ A community focused on connecting individuals to the importance of soil and its ecological functions.
  - _Heuristic:_ user-flagged for Aaron citation review (phonetic noise)
  - _Source:_ [ep 78 at 44:20] "The Y on Earth Community Stewardship and Sustainability Podcast Series is hosted by Erin William Perry, author, thought leader, and executive consultant."
  - _(member name not in transcript; likely Whisper drift — found canonical `Y on Earth Community` instead)_
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `Wieners Community` (mc=1, eps=[84])
  - _Description:_ A community initiative aimed at raising awareness about ecological restoration and sustainability efforts.
  - _Heuristic:_ via=fuzzy score=0.000 overlap=0.23
  - _Source:_ [ep 84 at 50:32] "to our audience and the why on earth community yeah I guess it's it's something that I noted in the"
  - _(member name not in transcript; likely Whisper drift — found canonical `Y on Earth Community` instead)_
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `Weiner Community` (mc=1, eps=[100])
  - _Description:_ An organization that provides resources and information related to biodynamics and sustainable agriculture.
  - _Heuristic:_ user-flagged for Aaron citation review (phonetic noise)
  - _Source:_ [ep 100 at 02:19] "And for those who don't know, Artem is on the board of directors of the Y on Earth community and has been tremendously integral to everything we've been doing the last few years."
  - _(member name not in transcript; likely Whisper drift — found canonical `Y on Earth Community` instead)_
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `Wine Community` (mc=1, eps=[161])
  - _Description:_ A collective of individuals and businesses involved in wine production and related activities, fostering collaboration and sustainability.
  - _Heuristic:_ user-flagged for Aaron citation review (phonetic noise)
  - _Source:_ [ep 161 at 40:00] "this is the why on earth community podcast I'm your host Aaron William Perry and today we're"
  - _(member name not in transcript; likely Whisper drift — found canonical `Y on Earth Community` instead)_
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `Y-O-Earth` (mc=1, eps=[151])
  - _Description:_ A community organization focused on environmental sustainability and education.
  - _Heuristic:_ via=fuzzy score=0.967 overlap=0.23
  - _Source:_ [ep 151 at 50:16] "Y-O-Earth community ambassadors, many of whom are giving on a monthly basis. And if you're not yet"
  - **Merge?** ☐ YES   ☐ NO

---

## `c0002` — Canonical: **YonEarth Community Stewardship and Sustainability Podcast** (EVENT)

_Canonical KG description:_ A podcast series focused on sustainability and community stewardship, featuring discussions with various thought leaders.

- **Member:** `Sustainability Podcast Series` (mc=6, eps=[59, 76, 93, 111, 119])
  - _Description:_ A podcast series hosted by YonEarth that discusses various topics related to sustainability and environmental stewardship.
  - _Heuristic:_ via=fuzzy+semantic score=0.000 overlap=0.45
  - _Source:_ [ep 59 at 00:06] "Welcome to the Y on Earth communities stewardship and sustainability podcast series."
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `Y on Earth community stewardship and sustainability podcast series` (mc=3, eps=[110, 115, 149])
  - _Description:_ A podcast series that discusses various topics related to sustainability and community engagement.
  - _Heuristic:_ via=fuzzy+semantic score=0.000 overlap=0.44
  - _Source:_ [ep 110 at 61:48] "to think about I wish you all many blessings the why on earth community stewardship and sustainability"
  - _(member name not verbatim; matched token `sustainability`)_
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `sustainability podcast series` (mc=2, eps=[104, 133])
  - _Description:_ A series of podcast episodes focused on topics related to sustainability and community engagement.
  - _Heuristic:_ via=fuzzy score=0.000 overlap=0.33
  - _Source:_ [ep 104 at 55:25] "The Y on Earth Community Stewardship and Sustainability podcast series is hosted by Erin William Perry"
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `Why On Earth Communities Stewardship and Sustainability Podcast Series` (mc=1, eps=[44])
  - _Description:_ A podcast series focused on sustainability and stewardship practices in communities.
  - _Heuristic:_ via=semantic score=0.000 overlap=0.44
  - _Source:_ [ep 44 at 00:08] "Welcome to the Writers' Community's Stewardship and Sustainability Podcast Series."
  - _(member name not verbatim; matched token `sustainability`)_
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `Why on Earth Community Stewardship and Sustainability Podcast Series` (mc=1, eps=[112])
  - _Description:_ A podcast series that discusses topics related to sustainability and community stewardship.
  - _Heuristic:_ via=fuzzy+semantic score=0.000 overlap=0.44
  - _Source:_ [ep 112 at 71:15] "the why on earth community stewardship and sustainability podcast series is hosted by"
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `The Y on Earth Community Stewardship and Sustainability Podcast` (mc=1, eps=[162])
  - _Description:_ A podcast series focused on topics related to sustainability and community stewardship.
  - _Heuristic:_ via=fuzzy+semantic score=0.000 overlap=0.44
  - _Source:_ [ep 162 at 70:33] "Absolutely. Likewise. Bye bye. Ciao. The Why on Earth Community Stewardship and Sustainability"
  - _(member name not verbatim; matched token `sustainability`)_
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `YonEarth Communities Stewardship and Sustainability Podcasts` (mc=1, eps=[66])
  - _Description:_ A podcast series dedicated to discussions on sustainability, climate action, and community mobilization.
  - _Heuristic:_ via=semantic score=0.935 overlap=0.44
  - _Source:_ [ep 66 at 00:07] "Welcome to the Y on Earth communities stewardship and sustainability podcast series."
  - _(member name not verbatim; matched token `sustainability`)_
  - **Merge?** ☐ YES   ☐ NO

---

## `c0004` — Canonical: **Lidge Family Foundation** (ORGANIZATION)

_Canonical KG description:_ A foundation that supports various initiatives, including those focused on sustainability and environmental stewardship.

- **Member:** `Lich Family Foundation` (mc=3, eps=[70, 105, 114])
  - _Description:_ A philanthropic organization supporting various initiatives, including those focused on sustainability and community health.
  - _Heuristic:_ via=fuzzy score=0.926 overlap=0.57
  - _Source:_ [ep 70 at 35:06] "productions, Patagonia, the Lich Family Foundation, Purium, and Weile Waters. And I also want to give"
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `Litch Family Foundation` (mc=1, eps=[88])
  - _Description:_ A philanthropic organization supporting various causes, including sustainability.
  - _Heuristic:_ via=fuzzy score=0.913 overlap=0.43
  - _Source:_ [ep 88 at 29:30] "Earth Coast productions, the Litch Family Foundation, Alpine Botanicals, Purium, Earth Hero,"
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `Ich Family Foundation` (mc=1, eps=[105])
  - _Description:_ A foundation that supports various initiatives, likely focused on sustainability and community development.
  - _Heuristic:_ via=fuzzy score=0.910 overlap=0.57
  - _Source:_ [ep 105 at 05:09] "the Lich Family Foundation, Alpine Botanicals, Purium, Earth Hero, Liquid Trainer, Vera Herbles,"
  - **Merge?** ☐ YES   ☐ NO

---

## `c0001` — Canonical: **YonEarth.org** (ORGANIZATION)

_Canonical KG description:_ A platform that supports community stewardship and sustainability initiatives, offering resources and support for individuals interested in these topics.

- **Member:** `whyoners.org` (mc=2, eps=[61, 67])
  - _Description:_ A website and platform dedicated to sharing information and resources related to soil regeneration and environmental stewardship.
  - _Heuristic:_ via=fuzzy score=0.000 overlap=0.27
  - _Source:_ [ep 61 at 24:30] "These sponsors are listed on the YonEarth.org-slaosh-support-page."
  - _(member name not in transcript; likely Whisper drift — found canonical `YonEarth.org` instead)_
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `yhonner.org` (mc=2, eps=[43, 118])
  - _Description:_ A website where individuals can join a monthly giving program to support sustainability initiatives.
  - _Heuristic:_ via=fuzzy score=0.000 overlap=0.36
  - _Source:_ [ep 43 at 54:42] "please visit yonearth.org backslash support support packages start at just one dollar per month"
  - _(member name not in transcript; likely Whisper drift — found canonical `YonEarth.org` instead)_
  - **Merge?** ☐ YES   ☐ NO

---

## `c0007` — Canonical: **Waylay Waters** (PRODUCT)

_Canonical KG description:_ CBD infused aroma therapy soaking salts offered as a thank you for joining the YonEarth community's monthly giving program.

- **Member:** `Wele Waters` (mc=5, eps=[117, 121, 165, 166, 168])
  - _Description:_ A social enterprise offering regenerative and biodynamically grown hemp-infused aromatherapy soaking salts.
  - _Heuristic:_ via=fuzzy score=0.000 overlap=0.25
  - _Source:_ [ep 117 at 23:30] "I'm your host Aaron William Perry and today we're visiting with the president of dr Bronners Mike Bronner and want to give a couple quick shout outs to some of our sponsors and partners this includes Wele Waters the r…"
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `Whaley Waters` (mc=4, eps=[92, 107, 151, 161])
  - _Description:_ Handmade aroma therapy soaking salts produced in Colorado, supporting the YonEarth Community's initiatives.
  - _Heuristic:_ via=fuzzy score=0.000 overlap=0.42
  - _Source:_ [ep 92 at 27:33] "And doing even more beyond that and that includes Purim Chelsea green the publisher Zen Bunny certified by dynamic coffee and they have chocolate to wail a waters which is our in house social enterprise where we're ma…"
  - _(member name not verbatim; matched token `waters`)_
  - **Merge?** ☐ YES   ☐ NO

---

## `c0023` — Canonical: **Whale Waters Soaking Salts** (PRODUCT)

_Canonical KG description:_ Hemp-infused aroma therapy soaking salts grown in Colorado, designed for relaxation and self-care.

- **Member:** `Weylay Water Soaking Salts` (mc=1, eps=[147])
  - _Description:_ A product made from hemp-infused aromatherapy soaking salts, produced by a social enterprise within the YonEarth community.
  - _Heuristic:_ via=fuzzy score=0.935 overlap=0.33
  - _Source:_ [ep 147 at 34:21] "greater. Weylay water soaking salts this is one of our social enterprises at the why on earth"
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `Weyley Water Soaking Salts` (mc=1, eps=[150])
  - _Description:_ Biodynamically and regeneratively grown hemp-infused aroma therapy soaking salts.
  - _Heuristic:_ via=fuzzy score=0.000 overlap=0.58
  - _Source:_ [ep 150 at 40:46] "Waley Waters Soaking Salts. These are the Colorado grown biodynamically and regeneratively grown hemp infused aroma therapy soaking salts we make for the Y on earth community."
  - _(member name not verbatim; matched token `soaking`)_
  - **Merge?** ☐ YES   ☐ NO

---

## `c0075` — Canonical: **Colorado State University** (ORGANIZATION)

_Canonical KG description:_ A public research university in Fort Collins, Colorado, known for its programs in agriculture, environmental science, and marine biology.

- **Member:** `Iowa State University` (mc=1, eps=[122])
  - _Description:_ Public research university where David Laird is a Professor Emeritus and has contributed to agronomy.
  - _Heuristic:_ via=fuzzy score=0.888 overlap=0.23
  - _Source:_ [ep 122 at 01:03] "PhD in Agronomy from Iowa State University in 1987. He is currently founder and president of N Sense"
  - **Merge?** ☐ YES   ☐ NO

- **Member:** `Ohio State University` (mc=1, eps=[71])
  - _Description:_ Public research university in Ohio known for its agricultural and environmental science programs.
  - _Heuristic:_ via=fuzzy score=0.000 overlap=0.54
  - _Source:_ [ep 71 at 01:13] "at Columbia University. Ms. Gore's previous experience includes serving as Director of Union"
  - _(member name not verbatim; matched token `university`)_
  - **Merge?** ☐ YES   ☐ NO

---

## `c0005` — Canonical: **Rodale Institute** (ORGANIZATION)

_Canonical KG description:_ An organization dedicated to improving the health of people and the planet through organic farming and regenerative practices.

- **Member:** `Rodeil Institute` (mc=1, eps=[68])
  - _Description:_ An organization involved in promoting regenerative agriculture and sustainability practices.
  - _Heuristic:_ via=fuzzy score=0.936 overlap=0.22
  - _Source:_ [ep 68 at 31:29] "was at MIT Masters choose its Institute of Technology in the 70s and he was already anticipating a"
  - _(member name not verbatim; matched token `institute`)_
  - **Merge?** ☐ YES   ☐ NO

---

## `c0010` — Canonical: **Wheylay Waters** (ORGANIZATION)

_Canonical KG description:_ A social enterprise that offers regeneratively grown and biodynamically grown hemp-infused aroma therapy soaking salts.

- **Member:** `Weyland Waters` (mc=1, eps=[158])
  - _Description:_ A company that produces biodynamically grown and infused aroma therapy soaking salts.
  - _Heuristic:_ via=fuzzy score=0.905 overlap=0.58
  - _Source:_ [ep 158 at 44:51] "and programming this includes of course Chelsea Green publishing Waylay and Waters the"
  - _(member name not verbatim; matched token `waters`)_
  - **Merge?** ☐ YES   ☐ NO

---

## `c0013` — Canonical: **Dr. Bronner's** (ORGANIZATION)

_Canonical KG description:_ A company known for its organic and fair trade personal care products, including soaps and hand sanitizers, committed to sustainable practices.

- **Member:** `Doctor Bronners` (mc=1, eps=[113])
  - _Description:_ A company known for its organic and fair trade personal care products.
  - _Heuristic:_ via=fuzzy score=0.000 overlap=0.50
  - _Source:_ [ep 113 at 21:00] "spring spa, earth water press, Dr. Bronners."
  - _(member name not verbatim; matched token `bronners`)_
  - **Merge?** ☐ YES   ☐ NO

---

## `c0031` — Canonical: **Association of Waldorf Schools of North America** (ORGANIZATION)

_Canonical KG description:_ An organization representing Waldorf education, which often emphasizes sustainability and environmental awareness.

- **Member:** `Association of Water Schools of North America` (mc=5, eps=[29, 45, 47, 59, 60])
  - _Description:_ An organization that promotes education and awareness about water sustainability in North America.
  - _Heuristic:_ via=fuzzy score=0.957 overlap=0.33
  - _Source:_ [ep 29 at 37:31] "this includes earth coast productions, wheylay waters, purium, the association of"
  - _(member name not verbatim; matched token `association`)_
  - **Merge?** ☐ YES   ☐ NO

---

## `c0035` — Canonical: **Mycelial Networks** (CONCEPT)

_Canonical KG description:_ Fungal networks in soil that serve as information superhighways, connecting plants and facilitating communication and nutrient exchange.

- **Member:** `Mycelium Network` (mc=1, eps=[95])
  - _Description:_ A network of fungal threads that connects plants and facilitates nutrient exchange, often used as a metaphor for interconnectedness in ecosystems.
  - _Heuristic:_ via=fuzzy score=0.000 overlap=0.31
  - _Source:_ [ep 95 at 21:20] "that we're creating an institution, it's that we're creating a network. And I would, I would call it"
  - _(member name not verbatim; matched token `network`)_
  - **Merge?** ☐ YES   ☐ NO

---

## `c0043` — Canonical: **Silver Pastures** (CONCEPT)

_Canonical KG description:_ A sustainable farming practice that integrates trees and pasture for livestock, promoting biodiversity and soil health.

- **Member:** `Silvo Pasture` (mc=1, eps=[124])
  - _Description:_ A land management practice that involves planting trees within pastures to enhance food production and habitat for livestock.
  - _Heuristic:_ via=fuzzy score=0.000 overlap=0.23
  - _Source:_ [ep 124 at 09:14] "And so at the same time we're breeding those drought tolerant silvo pasture and agroforestry crops for use on other projects as we continue to grow and expand which is pretty neat too."
  - **Merge?** ☐ YES   ☐ NO

---

## `c0067` — Canonical: **Homeowners Associations (HOAs)** (ORGANIZATION)

_Canonical KG description:_ Organizations in residential communities that manage common areas and enforce community rules, which can influence local environmental practices.

- **Member:** `HOA (Homeowners Association)` (mc=1, eps=[81])
  - _Description:_ A governing body in residential communities that makes and enforces rules for properties and residents.
  - _Heuristic:_ via=fuzzy score=0.912 overlap=0.23
  - _Source:_ [ep 81] _(name not in transcript verbatim; likely LLM extraction artifact)_  Episode: "Episode 81 – Lem Tingley, Chief Growing Officer, Growing Spaces Greenhouses"
  - **Merge?** ☐ YES   ☐ NO

---

## `c0082` — Canonical: **tree planting** (PRACTICE)

_Canonical KG description:_ The act of planting trees to benefit the environment and communities, often used as a method for humanitarian aid and ecological restoration.

- **Member:** `Planting Trees` (mc=1, eps=[58])
  - _Description:_ The act of planting trees to enhance the environment, improve air quality, and support biodiversity.
  - _Heuristic:_ via=fuzzy score=0.976 overlap=0.27
  - _Source:_ [ep 58 at 04:35] "Bees, trees, soil and water."
  - _(member name not verbatim; matched token `trees`)_
  - **Merge?** ☐ YES   ☐ NO

---

## `c0083` — Canonical: **EarthX Film** (EVENT)

_Canonical KG description:_ An environmental festival that showcases films and interactive media related to conservation and climate change.

- **Member:** `Earth X Film Festival` (mc=1, eps=[130])
  - _Description:_ An environmental film festival held in Dallas, Texas, showcasing films that highlight environmental issues and local initiatives.
  - _Heuristic:_ via=fuzzy score=0.000 overlap=0.25
  - _Source:_ [ep 130 at 29:30] "all filmmakers who would like to exhibit. At the Earth X Film Festival we have a whole screening"
  - **Merge?** ☐ YES   ☐ NO

---

## `c0089` — Canonical: **Judith Schwartz** (PERSON)

_Canonical KG description:_ Author of 'The Reindeer Chronicles', known for her work on ecological restoration and sustainability.

- **Member:** `Judith D. Schwartz` (mc=1, eps=[84])
  - _Description:_ Author and speaker known for her work on ecological restoration and the importance of soil health.
  - _Heuristic:_ via=fuzzy score=0.967 overlap=0.56
  - _Source:_ [ep 84 at 00:09] "Hi friends, I'm so excited to share this episode with you. Our conversation is with Judith Schwartz."
  - _(member name not verbatim; matched token `schwartz`)_
  - **Merge?** ☐ YES   ☐ NO

---

## `c0091` — Canonical: **Biodiversity Loss** (CONCEPT)

_Canonical KG description:_ The decline in the variety of life in a particular habitat or ecosystem, often exacerbated by human activities.

- **Member:** `loss of biodiversity` (mc=1, eps=[1])
  - _Description:_ The decline in the variety of life on Earth, including the extinction of species and loss of ecosystems, often driven by human activities.
  - _Heuristic:_ via=fuzzy score=0.950 overlap=0.46
  - _Source:_ [ep 1 at 05:51] "and the loss of biodiversity and some of the other major threats to the planet. So I would say what"
  - **Merge?** ☐ YES   ☐ NO

---

## `c0114` — Canonical: **Growing Dome** (PRODUCT)

_Canonical KG description:_ A geodesic dome designed for growing plants year-round in various climates, promoting sustainable agriculture.

- **Member:** `Grow Domes` (mc=1, eps=[79])
  - _Description:_ Greenhouse structures designed to create optimal growing conditions for plants year-round.
  - _Heuristic:_ via=fuzzy score=0.000 overlap=0.45
  - _Source:_ [ep 79 at 04:23] "discovered growing spaces, the company that makes these grow domes which were sitting in"
  - **Merge?** ☐ YES   ☐ NO

---

## `c0118` — Canonical: **American Medical Association** (ORGANIZATION)

_Canonical KG description:_ A professional association and lobbying group of physicians and medical students in the United States.

- **Member:** `American Bar Association` (mc=1, eps=[51])
  - _Description:_ A professional association for lawyers and legal professionals in the United States, with various sections including international law.
  - _Heuristic:_ via=fuzzy score=0.901 overlap=0.36
  - _Source:_ [ep 51 at 00:54] "Section of the American Bar Association, Bawa Muayyadin Fellowship, Universal Sufi Council,"
  - **Merge?** ☐ YES   ☐ NO

---

## `c0121` — Canonical: **Oklahoma State University** (ORGANIZATION)

_Canonical KG description:_ University where Rowdy Yeatts graduated with a business degree.

- **Member:** `Sonoma State University` (mc=1, eps=[137])
  - _Description:_ A public university in California where Georgia Kelly obtained a certificate in conflict resolution.
  - _Heuristic:_ via=fuzzy score=0.886 overlap=0.20
  - _Source:_ [ep 137 at 02:24] "Georgia also holds a certificate in conflict resolution from Sonoma State University"
  - **Merge?** ☐ YES   ☐ NO

---

## `c0125` — Canonical: **economic inequality** (CONCEPT)

_Canonical KG description:_ The disparity in wealth and income distribution among individuals in a society, which Mondragon aims to reduce.

- **Member:** `income inequality` (mc=1, eps=[12])
  - _Description:_ The unequal distribution of income within a population, which can drive social and economic challenges related to sustainability.
  - _Heuristic:_ via=fuzzy score=0.926 overlap=0.25
  - _Source:_ [ep 12 at 26:07] "on the social side, there are issues with increasing income inequality and lack of trust and cooperation."
  - **Merge?** ☐ YES   ☐ NO

---

## `c0141` — Canonical: **Autonomous Vehicles** (TECHNOLOGY)

_Canonical KG description:_ Self-driving cars that use technology to navigate and operate without human intervention.

- **Member:** `Self-Driving Cars` (mc=1, eps=[94])
  - _Description:_ Autonomous vehicles that use technology to navigate and drive without human intervention.
  - _Heuristic:_ via=semantic score=0.938 overlap=0.56
  - _Source:_ [ep 94 at 01:25] "experience at Google X, developing technology such as Google Glass and Google's self-driving cars."
  - **Merge?** ☐ YES   ☐ NO

---

## `c0146` — Canonical: **Chelsea Green Publishing** (ORGANIZATION)

_Canonical KG description:_ A publishing company known for its focus on sustainable living and environmental topics, including books on regenerative agriculture.

- **Member:** `Chelsea Green Publishers` (mc=1, eps=[127])
  - _Description:_ A publishing company focused on books about sustainable living, environmental issues, and social justice.
  - _Heuristic:_ via=fuzzy score=0.917 overlap=0.45
  - _Source:_ [ep 127 at 33:47] "force, agmatic, Chelsea Green publishers, organic India, and again, Walei Waters. You can go"
  - **Merge?** ☐ YES   ☐ NO

---

## `c0179` — Canonical: **Industrial Agriculture** (CONCEPT)

_Canonical KG description:_ A modern form of agriculture that involves the intensive production of crops and livestock, often characterized by the use of chemical fertilizers and pesticides.

- **Member:** `Industrialized Agriculture` (mc=1, eps=[114])
  - _Description:_ A system of farming characterized by large-scale production and reliance on chemical inputs, often leading to soil degradation.
  - _Heuristic:_ via=fuzzy score=0.949 overlap=0.31
  - _Source:_ [ep 114 at 02:38] "field of top soil is blowing away and being destroyed due to industrialized agriculture."
  - **Merge?** ☐ YES   ☐ NO

---

## `c0230` — Canonical: **Nature Connection** (CONCEPT)

_Canonical KG description:_ The practice of fostering a deep relationship with the natural world, often leading to increased well-being and awareness.

- **Member:** `connection with nature` (mc=2, eps=[1, 67])
  - _Description:_ The act of developing a deeper relationship with the natural environment, which can enhance well-being and promote sustainability.
  - _Heuristic:_ via=fuzzy score=0.924 overlap=0.27
  - _Source:_ [ep 1 at 18:48] "connection with nature we're working with well-being practices and it's so exciting to see with"
  - **Merge?** ☐ YES   ☐ NO

---

## `c0251` — Canonical: **Ganges River** (PLACE)

_Canonical KG description:_ A major river in India, considered sacred in Hinduism, where many people go for spiritual and health-related practices.

- **Member:** `River Ganges` (mc=1, eps=[118])
  - _Description:_ A sacred river in India, significant in Hindu tradition, where various rituals and ceremonies are performed.
  - _Heuristic:_ via=fuzzy score=1.000 overlap=0.31
  - _Source:_ [ep 118 at 50:42] "and it's called the marriage of Tulsi even though it refers to the marriage of Lakshmi to Vishnu right but they call it the marriage of Tulsi and part of the tradition there is the mothers take Tulsi plants down to th…"
  - **Merge?** ☐ YES   ☐ NO

---

## `c0397` — Canonical: **Soil Aggregates** (CONCEPT)

_Canonical KG description:_ Clusters of soil particles that improve soil structure, water retention, and aeration, essential for healthy plant growth.

- **Member:** `Aggregated Soil` (mc=1, eps=[91])
  - _Description:_ Soil that is clumped together in aggregates, which improves its structure, aeration, and water retention.
  - _Heuristic:_ via=fuzzy score=0.932 overlap=0.42
  - _Source:_ [ep 91 at 20:13] "humus or every piece of aggregated soil those phenomenons that nature evolved over millions and"
  - **Merge?** ☐ YES   ☐ NO

---

## `c0451` — Canonical: **Carbon to Nitrogen Ratio** (CONCEPT)

_Canonical KG description:_ A measure used in soil science to describe the balance of carbon and nitrogen in organic matter, influencing microbial activity and soil health.

- **Member:** `Carbon Nitrogen Ratio` (mc=1, eps=[103])
  - _Description:_ The ratio of carbon to nitrogen in organic materials, important for composting and energy production efficiency.
  - _Heuristic:_ via=fuzzy score=0.958 overlap=0.21
  - _Source:_ [ep 103 at 16:10] "need one of 30 to one carbon nitrogen ratio. Just stuff we're putting in here. But most maneuvers are"
  - **Merge?** ☐ YES   ☐ NO

---

## `c0465` — Canonical: **Sixth Grade Extinction** (CONCEPT)

_Canonical KG description:_ A term referring to the ongoing extinction event caused by human activities, leading to significant loss of biodiversity.

- **Member:** `sixth great extinction` (mc=1, eps=[144])
  - _Description:_ A term used to describe the ongoing extinction event caused by human activity, threatening a large percentage of species.
  - _Heuristic:_ via=fuzzy score=0.922 overlap=0.46
  - _Source:_ [ep 144 at 08:03] "like potentially the sixth grade extinction. Yeah. We're observing in geolotics."
  - _(member name not verbatim; matched token `extinction`)_
  - **Merge?** ☐ YES   ☐ NO

---

## `c0485` — Canonical: **Regenerative Organic Farming** (CONCEPT)

_Canonical KG description:_ An agricultural approach that integrates regenerative practices to improve soil health, biodiversity, and ecosystem resilience.

- **Member:** `Organic Regenerative Farming` (mc=1, eps=[142])
  - _Description:_ A farming approach that emphasizes organic practices and regenerative techniques to restore soil health and ecosystems.
  - _Heuristic:_ via=fuzzy score=1.000 overlap=0.45
  - _Source:_ [ep 142 at 01:16] "Throughout her career, she is advocated for the potential of organic regenerative farming"
  - **Merge?** ☐ YES   ☐ NO

---

## `c0494` — Canonical: **Biodynamic Soil Preps** (PRACTICE)

_Canonical KG description:_ Preparations used in biodynamic agriculture to enhance soil health and ecology.

- **Member:** `Biodynamic Preps` (mc=1, eps=[116])
  - _Description:_ Preparations used in biodynamic agriculture that enhance soil health and plant growth, often made from natural materials.
  - _Heuristic:_ via=fuzzy score=0.921 overlap=0.54
  - _Source:_ [ep 116] _(name not in transcript verbatim; likely LLM extraction artifact)_  Episode: "David Sandoval Episode 116 – David Sandoval, Co-Founder, Purium Organic Superfood Company"
  - **Merge?** ☐ YES   ☐ NO

---

## `c0496` — Canonical: **Dr. Bronner's Magic Chocolate** (PRODUCT)

_Canonical KG description:_ A chocolate product associated with Dr. Bronner, known for its organic and fair trade ingredients.

- **Member:** `Dr. Bronner’s chocolate` (mc=1, eps=[117])
  - _Description:_ A brand of chocolate known for its ethical sourcing and commitment to fair trade practices.
  - _Heuristic:_ via=fuzzy score=0.929 overlap=0.44
  - _Source:_ [ep 117 at 00:22] "Hey, how's it going? Great. How are you doing today? Oh really well. Happy to be here in sunny Southern California. Yeah, excellent. And I'm especially excited about our conversation today because we're going to be ta…"
  - _(member name not verbatim; matched token `chocolate`)_
  - **Merge?** ☐ YES   ☐ NO

---

## `c0507` — Canonical: **Crop Residues** (PRODUCT)

_Canonical KG description:_ The leftover plant materials after harvest, which can be used as feedstock for producing biochar.

- **Member:** `Corn Residues` (mc=1, eps=[120])
  - _Description:_ Leftover materials from corn crops that can be utilized as feedstock for biochar.
  - _Heuristic:_ via=fuzzy score=0.921 overlap=0.40
  - _Source:_ [ep 120 at 09:42] "different crop residues, outholes, corn residues, different things like that that might make a"
  - **Merge?** ☐ YES   ☐ NO

---

## `c0508` — Canonical: **biocharco.op** (ORGANIZATION)

_Canonical KG description:_ Website associated with biochar, likely providing resources or community engagement related to biochar technologies.

- **Member:** `hp biochar` (mc=1, eps=[120])
  - _Description:_ Online presence for High Plains biochar, providing information and resources related to their biochar products.
  - _Heuristic:_ via=fuzzy score=0.883 overlap=0.40
  - _Source:_ [ep 120 at 22:52] "Facebook at HP Biochar and of course invite you to check out some of the other related podcast"
  - **Merge?** ☐ YES   ☐ NO

---

## `c0517` — Canonical: **Wyon Earth Community Podcast** (EVENT)

_Canonical KG description:_ A podcast focused on sustainability and environmental issues, featuring various guests and discussions.

- **Member:** `YODE Earth Community Podcast` (mc=1, eps=[135])
  - _Description:_ A podcast series focused on sustainability and community action.
  - _Heuristic:_ via=fuzzy score=0.952 overlap=0.25
  - _Source:_ [ep 135] _(name not in transcript verbatim; likely LLM extraction artifact)_  Episode: "Episode 135 – Chief Tuwe & Ari Brasil – from the Heart of the Amazon"
  - **Merge?** ☐ YES   ☐ NO

---

## `c0568` — Canonical: **Rhineland Mystic movement** (CONCEPT)

_Canonical KG description:_ A spiritual and philosophical movement in the Rhineland region during the medieval period, characterized by mystical experiences and visions.

- **Member:** `Rhine Mystic Movement` (mc=1, eps=[134])
  - _Description:_ A spiritual movement associated with mystics from the Rhine region, emphasizing a deep connection to nature and the cosmos.
  - _Heuristic:_ via=fuzzy score=0.947 overlap=0.25
  - _Source:_ [ep 134 at 22:44] "Rhineland Mystic movement and of course the Rhine River runs from Switzerland down to the"
  - _(member name not verbatim; matched token `movement`)_
  - **Merge?** ☐ YES   ☐ NO

---

## `c0610` — Canonical: **Conference in February** (EVENT)

_Canonical KG description:_ An annual gathering where participants discuss and present texts related to spiritual and consciousness topics.

- **Member:** `February Conference` (mc=1, eps=[152])
  - _Description:_ An upcoming conference where participants can meet and discuss various topics related to the eco movement.
  - _Heuristic:_ via=fuzzy score=0.955 overlap=0.45
  - _Source:_ [ep 152 at 17:35] "conference we have here at the Guadianum where our big hole has 1000 places so we hope that we will be"
  - _(member name not verbatim; matched token `conference`)_
  - **Merge?** ☐ YES   ☐ NO

---

## `c0643` — Canonical: **pharmaceutical industries** (ORGANIZATION)

_Canonical KG description:_ Companies involved in the development, production, and marketing of medications, often critiqued for their profit-driven motives.

- **Member:** `pharmaceutical industry` (mc=1, eps=[90])
  - _Description:_ An industry focused on the development, production, and marketing of medications.
  - _Heuristic:_ via=fuzzy score=0.885 overlap=0.36
  - _Source:_ [ep 90 at 21:36] "And it's interesting to me too that the alchemical origins of what we now call modern chemistry or even biochemistry and bio-pharmacology, you know, that is the pharmaceutical industry."
  - **Merge?** ☐ YES   ☐ NO

---

## `c0653` — Canonical: **Democratic Republic of the Congo** (PLACE)

_Canonical KG description:_ A country in Central Africa that has faced prolonged conflict and challenges in maintaining banking operations and community values.

- **Member:** `Democratic Republic of Congo` (mc=1, eps=[163])
  - _Description:_ A country in Central Africa mentioned as having a member bank of the Global Alliance for Banking on Values.
  - _Heuristic:_ via=fuzzy score=0.958 overlap=0.45
  - _Source:_ [ep 163 at 34:23] "democratic republican of Congo I believe there's one uh in the general area of Russia and"
  - _(member name not verbatim; matched token `democratic`)_
  - **Merge?** ☐ YES   ☐ NO

---

## `c0657` — Canonical: **Ring of Fire Biochar Kiln** (PRODUCT)

_Canonical KG description:_ A specific type of kiln designed for producing biochar efficiently.

- **Member:** `Ring of Fire Kiln` (mc=1, eps=[165])
  - _Description:_ A type of kiln used for producing biochar, designed to facilitate the conversion of organic material into charcoal.
  - _Heuristic:_ via=semantic score=0.921 overlap=0.42
  - _Source:_ [ep 165 at 05:17] "Got it. That's great. Yeah. And of course, BTUs being British thermal units, a way of measuring heat capacity or potential color potential in fuels when we combust them, right?"
  - _(member name not verbatim; matched token `ring`)_
  - **Merge?** ☐ YES   ☐ NO

---

## `c0690` — Canonical: **Judeo-Christian creation stories** (CONCEPT)

_Canonical KG description:_ Religious narratives from Judeo-Christian traditions that describe the creation of the world and humanity, often viewed in a linear time framework.

- **Member:** `Judeo-Christian creation story` (mc=1, eps=[58])
  - _Description:_ A narrative from the Judeo-Christian tradition that describes the creation of the world, emphasizing the role of light.
  - _Heuristic:_ via=fuzzy score=0.958 overlap=0.29
  - _Source:_ [ep 58] _(name not in transcript verbatim; likely LLM extraction artifact)_  Episode: "Episode 58 – Sarah Drew, Visionary Author, Gaia Codex"
  - **Merge?** ☐ YES   ☐ NO

---

## `c0702` — Canonical: **Recyclable Materials** (PRODUCT)

_Canonical KG description:_ Materials that can be processed and reused, which Beauty Counter is prioritizing in their packaging efforts.

- **Member:** `Recycled Materials` (mc=1, eps=[90])
  - _Description:_ Materials that have been processed and repurposed from waste to create new products, promoting sustainability.
  - _Heuristic:_ via=fuzzy score=0.931 overlap=0.22
  - _Source:_ [ep 90 at 61:24] "and chasing all these materials, that's what's firing off."
  - _(member name not verbatim; matched token `materials`)_
  - **Merge?** ☐ YES   ☐ NO

---

## `c0725` — Canonical: **Winner’s Community Podcast** (EVENT)

_Canonical KG description:_ A podcast series focused on sustainability, technology, and community engagement.

- **Member:** `Writers Community podcast` (mc=1, eps=[78])
  - _Description:_ A podcast that features discussions on various topics, including sustainability and environmental issues.
  - _Heuristic:_ via=fuzzy score=0.920 overlap=0.22
  - _Source:_ [ep 78] _(name not in transcript verbatim; likely LLM extraction artifact)_  Episode: "Episode 78 – Maria Nikulkova, Mysterious Microbiology"
  - **Merge?** ☐ YES   ☐ NO

---

## `c0727` — Canonical: **Mesoamerican Rift** (PLACE)

_Canonical KG description:_ Geological feature in Central America recognized for its significant biodiversity.

- **Member:** `Mesoamerican Reef` (mc=1, eps=[80])
  - _Description:_ A coral reef system in the Caribbean, recognized for its significant biodiversity.
  - _Heuristic:_ via=fuzzy score=0.922 overlap=0.43
  - _Source:_ [ep 80 at 27:12] "North of Mexico to the American, Mesoamerican reef that actually represent almost the 15% of biodiversity in the world."
  - **Merge?** ☐ YES   ☐ NO

---

## `c0736` — Canonical: **Singaporean government** (ORGANIZATION)

_Canonical KG description:_ The governing body of Singapore, noted for its compassionate and humanistic approach to economic management.

- **Member:** `Norwegian government` (mc=1, eps=[84])
  - _Description:_ The governing body of Norway, involved in land management and resource extraction policies affecting indigenous lands.
  - _Heuristic:_ via=fuzzy score=0.885 overlap=0.25
  - _Source:_ [ep 84 at 43:20] "yes Indigenous knowledge and it was to commemorate the 100th anniversary of the Norwegian government at"
  - **Merge?** ☐ YES   ☐ NO

---

## `c0744` — Canonical: **Post-Colonial Economics** (CONCEPT)

_Canonical KG description:_ An economic framework that critiques traditional economic theories and practices that emerged from colonial histories and emphasizes alternative approaches.

- **Member:** `neo-colonial economics` (mc=1, eps=[92])
  - _Description:_ A term describing economic practices that perpetuate colonial patterns of exploitation and inequality.
  - _Heuristic:_ via=fuzzy score=0.926 overlap=0.25
  - _Source:_ [ep 92 at 00:15] "Today we're doing a special Thanksgiving holiday shopping and regenerative economics episode."
  - _(member name not verbatim; matched token `economics`)_
  - **Merge?** ☐ YES   ☐ NO

---

