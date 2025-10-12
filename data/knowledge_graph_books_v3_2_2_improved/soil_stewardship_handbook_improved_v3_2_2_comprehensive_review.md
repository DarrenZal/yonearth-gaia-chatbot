# Comprehensive Knowledge Graph Extraction Review

**Extraction File**: `data/knowledge_graph_books_v3_2_2_improved/soil_stewardship_handbook_improved_v3_2_2.json`
**Source Book**: `Soil-Stewardship-Handbook-eBook.pdf`
**Total Pages in Book**: 53
**Pages with Extractions**: 12
**Total Relationships**: 170

## Executive Summary

- **Incorrect Relationships**: 9 out of 170 (5.3%)
- **Pages Fully Skipped**: 41 pages with no extractions
- **Pages with Missing Knowledge**: 35 pages likely missing content
- **Coverage**: 22.6% of pages have at least one extraction

## Part 1: Incorrect Relationships

Found **9** incorrect relationships:

### 1. Incorrect Relationship

**Triple**: `soil` → `is the foundation of` → `human life`

**Page**: 15

**Evidence**:
> SOIL—THE FOUNDATION OF HUMAN LIFE “Ultimately, the only wealth that can sustain any community, economy or nation is derived from the photosynthetic process—green plants growing on regenerating soil.” —Allan Savory...

**Issue**: Missing qualifiers from evidence: synthetic

---

### 2. Incorrect Relationship

**Triple**: `soil` → `enhances` → `our immune systems`

**Page**: 18

**Evidence**:
> Soil heals us. By getting our hands in the living “dirt,” we literally soothe the anxieties of daily stress, enhance our immune systems, and increase our production of serotonin....

**Issue**: Missing qualifiers from evidence: daily

---

### 3. Incorrect Relationship

**Triple**: `soil` → `increases` → `our production of serotonin`

**Page**: 18

**Evidence**:
> Soil heals us. By getting our hands in the living “dirt,” we literally soothe the anxieties of daily stress, enhance our immune systems, and increase our production of serotonin....

**Issue**: Missing qualifiers from evidence: daily

---

### 4. Incorrect Relationship

**Triple**: `living soil` → `makes us feel better` → `us`

**Page**: 18

**Evidence**:
> Our physical connection with living soil literally makes us feel better and makes us smarter!...

**Issue**: Entity too generic/vague: 'living soil' or 'us'

---

### 5. Incorrect Relationship

**Triple**: `restoring agricultural soils` → `will enhance and restore` → `the health—physical, mental and emotional—of humanity`

**Page**: 18

**Evidence**:
> By restoring agricultural soils to their natural, organic, and productively vital states, we will also enhance and restore the health—physical, mental and emotional—of humanity....

**Issue**: Missing qualifiers from evidence: organic, natural

---

### 6. Incorrect Relationship

**Triple**: `human activity` → `has increased` → `the amount of carbon in the atmosphere by over 40%`

**Page**: 19

**Evidence**:
> our human activity has increased the amount of carbon in the atmosphere by over 40%!...

**Issue**: Target 'the amount of carbon in the atmosphere by over 40%' is a number/percentage but source 'human activity' lacks measurement context (should specify what is being measured) | Semantically odd: 'human activity has increased the amount of carbon in the atmosphere by over 40%' doesn't make sense

---

### 7. Incorrect Relationship

**Triple**: `we` → `can reverse` → `climate change by collaborating with Earth’s living systems to rebuild soil`

**Page**: 19

**Evidence**:
> we can reverse climate change by collaborating with Earth’s living systems to rebuild soil....

**Issue**: Entity too generic/vague: 'we' or 'climate change by collaborating with Earth’s living systems to rebuild soil'

---

### 8. Incorrect Relationship

**Triple**: `soil carbon` → `can increase by` → `10%`

**Page**: 21

**Evidence**:
> we’re only talking about an increase of soil carbon of about 10%. It’s not necessarily easy to do....

**Issue**: Target '10%' is a number/percentage but source 'soil carbon' lacks measurement context (should specify what is being measured) | Semantically odd: 'soil carbon can increase by 10%' doesn't make sense

---

### 9. Incorrect Relationship

**Triple**: `Stephanie Held` → `authored` → `10 Detroit Urban Farms Rooting Goodness Into the City`

**Page**: 39

**Evidence**:
> Held, Stephanie. “10 Detroit Urban Farms Rooting Goodness Into the City.” Daily Detroit. July 6, 2015....

**Issue**: Missing qualifiers from evidence: daily

---

## Part 2: Missing Knowledge by Page

Found **35** pages with likely missing extractions:

### Page 4

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (definitions, processes, solutions, locations) but NO relationships were extracted

**Text Sample**:
```
“All life starts and ends with earth. The Soil Stewardship Handbook looks at our connections to the
soil and the way that relationship can affect so many aspects of life. A cool resource and a creative
perspective on a subject all too often not considered.”
—Tanner Watt
Director of Partnership and Development, REVERB
“The Soil Stewardship Handbook is timely and empowering. The health of a communit
...```

---

### Page 5

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (definitions, locations) but NO relationships were extracted

**Text Sample**:
```
Copyright © 2018 Aaron William Perry
All rights reserved. No part of this book may be reproduced in any form or by any electronic or mechanical means,
including information storage and retrieval systems, without permission in writing from the publisher, except by
reviewers, who may quote brief passages in a review.
First Printing January 2018
Printed in the United States of America
Soil Stewardshi
...```

---

### Page 6

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (definitions, locations) but NO relationships were extracted

**Text Sample**:
```
This book is dedicated to my two children, Osha and Hunter, whose brilliance,
courage, determination and compassion give me great hope for the future.
This book is also dedicated to all other children alive today on Earth—of all ages—
and the future generations who will follow us.
And this book is dedicated to the Y on Earth Community, the Soil Stewardship
Guild members and the Community Impact Am
...```

---

### Page 8

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (organizations, locations) but NO relationships were extracted

**Text Sample**:
```
CONTENTS
Forward ix
Note from the Author xi
Soil Stewardship 1
Why Soil Stewardship? 1
What Can We Do? 2
Soil—The Foundation of Human Life 4
Soil—Enhancing Intelligence, Health and Well-Being 5
Soil—Healing Earth and Restoring Balance 6
Soil—Reversing Climate Change 7
Soil-Building Explained: Practical and Awesome! 9
Soil Stewardship—A Journey to Mastery 10
Soil—An Alchemy of Love 14
Envision 16
S
...```

---

### Page 10

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (definitions, processes, problems, solutions, locations) but NO relationships were extracted
- Page has 6 knowledge indicators but only 0 relationships extracted - likely missing content

**Text Sample**:
```
FORWARD
I hail from the verdant landscape of Europe’s eastern Alpine region, where my people have lived
for centuries in what is today known as Slovenia. Our land of glorious mountains, mysterious
caverns, glimmering lakes, rolling hillocks and dancing meadows has been a cultural crossroads
for millennia, and has in many ways remained a veritable Eden through the ages. In times of both
peace and t
...```

---

### Page 11

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (definitions, locations) but NO relationships were extracted

**Text Sample**:
```
x AARON WILLIAM PERRY
We have the choice to thrive and to heal—ourselves, our communities, and our planet—by
connecting with the living soil, and by cultivating a deep awareness of the awesome miracle that
is life on Earth. Respect and reverence will surely follow—which are necessary to take good care
of this wonderful place upon which all of our lives depend.
I hope you’ll roll up your sleeves, d
...```

---

### Page 14

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (solutions, locations) but NO relationships were extracted

**Text Sample**:
```
SOIL STEWARDSHIP
“Find your place on the planet. Dig in, and take responsibility from there.”
—Gary Snyder
“Soil is the answer. We are asking many questions on this journey together.
Questions about our lives, our health and well-being, and about the
sustainability of our planet. We will discover that soil is the answer
to so many of these questions.”
—Y on Earth
“What we do to the soil, we do to 
...```

---

### Page 16

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (definitions, processes, causes, locations) but NO relationships were extracted
- Page contains lists but only 0 relationships extracted

**Text Sample**:
```
SOIL STEWARDSHIP HANDBOOK 3
HERE IS THE BASIC FRAMEWORK—IT’S SO EASY TO GET STARTED!
APPRENTICE (BEGINNER) LEVEL
COMPOST is a nutrient-rich and biologically
• Compost
vibrant soil amendment produced through
• Grow House Plants the natural decomposition of food waste,
• Buy Food, Beverage and Clothing leaves, grass clippings and other plant-based
Products with Soil Stewardship in materials. The pro
...```

---

### Page 17

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (processes, organizations, locations) but NO relationships were extracted

**Text Sample**:
```
4 AARON WILLIAM PERRY
SOIL—THE FOUNDATION OF HUMAN LIFE
“Ultimately, the only wealth that can sustain any community, economy
or nation is derived from the photosynthetic process—green plants
growing on regenerating soil.”
—Allan Savory
We humans—our humanity—are so inextricably linked to soil, that describing this profound
interconnectivity nearly defi es words. It is strange that something so dea
...```

---

### Page 20

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (definitions, statistics, processes, locations) but NO relationships were extracted
- Page has 7 knowledge indicators but only 0 relationships extracted - likely missing content

**Text Sample**:
```
SOIL STEWARDSHIP HANDBOOK 7
Right now, too much of our chemical agriculture is doing just the opposite.
But we can change all of this! Here’s the thing—the good news, the hopeful truth: when properly
treated and cared for, soil is a renewable resource!
We have so many opportunities to heal living soil—from large-scale, multi-national projects like
the Great Green Wall of the southern Sahara region
...```

---

### Page 21

**Relationships Extracted**: 3

**Issues**:
- Page contains lists but only 3 relationships extracted

**Text Sample**:
```
8 AARON WILLIAM PERRY
reduction of atmospheric carbon,
Earth Carbon Balance*
PPM Billion Tons % Total
which is achieved by soil-building.
Atmosphere Now 405 8 00 21%
We will reverse climate change by
re-carbonizing soil—all around Soil Now n/a 2,500 65%
planet Earth! Th e amount of fossil Flora & Fauna Now n/a 5 60 15%
carbon that we need to return to the TOTAL 3,860 100%
ground is an amount equal
...```

---

### Page 22

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (definitions, statistics, processes, locations) but NO relationships were extracted
- Page has 6 knowledge indicators but only 0 relationships extracted - likely missing content

**Text Sample**:
```
SOIL STEWARDSHIP HANDBOOK 9
SOIL-BUILDING EXPLAINED: PRACTICAL
AND AWESOME!
But what does it mean, exactly, to build soil? Soil building is a natural process, a continuous
cycle that has been in motion for hundreds of millions of years on Earth. Because we humans
have destroyed so much soil, and have emitted so much fossil carbon into the atmosphere, it is
imperative that we collaborate with natur
...```

---

### Page 23

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (definitions, processes, solutions, people, locations) but NO relationships were extracted
- Page has 6 knowledge indicators but only 0 relationships extracted - likely missing content

**Text Sample**:
```
10 AARON WILLIAM PERRY
SOIL STEWARDSHIP—A JOURNEY TO MASTERY
Th e process that turns garbage into a garden is central to our survival. We
depend on dirt to purify and heal the systems that sustain us.
—Peter Girguis
We start with composting. Th is “closing of the loop” of organic nutrients is a critical and necessary
starting point. Th ose peels and scraps from preparing dinner? Soon to be soil! T
...```

---

### Page 24

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (definitions, processes, problems, locations) but NO relationships were extracted
- Page contains lists but only 0 relationships extracted

**Text Sample**:
```
SOIL STEWARDSHIP HANDBOOK 11
organisms, and locks up carbon in the soil that was just in the atmosphere a few years or even
months prior. Biochar is key to putting climate-changing greenhouse gases from generations of
fossil fuel emissions back down into the ground where it belongs!
But this isn’t just about the direct benefi ts to humanity through the
regeneration and stewardship of soil. Th is i
...```

---

### Page 25

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (definitions, benefits, locations) but NO relationships were extracted

**Text Sample**:
```
12 AARON WILLIAM PERRY
3. STEWARD THROUGH CHOICE (Apprentice Level)—Every time we buy a food item,
beverage, or piece of clothing, we are directly impacting soil somewhere in the world. Often
dozens of times per day! Choose organic, biodynamic, and soil-regenerating products—
they’re better for you and for our planet Earth! (Used and recycled clothing are a great option
too!).
4. PLANT TREES (Prac
...```

---

### Page 26

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (processes, solutions, organizations, locations) but NO relationships were extracted
- Page contains lists but only 0 relationships extracted

**Text Sample**:
```
SOIL STEWARDSHIP HANDBOOK 13
12. BECOME A SOIL STEWARDSHIP GUILD AMBASSADOR (Master Level)—Help others get
started and succeed on their journey to soil stewardship mastery! Doing so will help reinforce
your quest, your own learning and your own practice, and your humble leadership will be of
great value to your community.
SOIL STEWARDSHIP PRACTICES FOR COMMUNITY
“Th e regeneration of our soil is t
...```

---

### Page 27

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (definitions, statistics, processes, organizations, locations) but NO relationships were extracted
- Page has 7 knowledge indicators but only 0 relationships extracted - likely missing content

**Text Sample**:
```
14 AARON WILLIAM PERRY
7. ORGANIZE SOIL BUILDING FLASH MOB PARTIES—Like Soil Building Parties, these Flash
Mob events are all about getting together to build soil—but are a bit more theatrical and intense
in seeking to accomplish a whole lot in a relatively short period of time. Th ese are equal parts
coordination and perspiration and will be an excellent cause for celebration once completed!
SOIL
...```

---

### Page 28

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (definitions, statistics, processes, locations) but NO relationships were extracted

**Text Sample**:
```
SOIL STEWARDSHIP HANDBOOK 15
our minds, bodies and spirits—into balance and wholeness. Thus we will cultivate and embody a
humble human holiness, directly connected with the vibrating life force that animates the entire
universe and all life on Earth.
There is a very real and powerful alchemy in
soil. Many of our ancient traditions speak of
the elements: earth, air, fire and water. Where
else do w
...```

---

### Page 29

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (statistics, processes, locations) but NO relationships were extracted

**Text Sample**:
```
16 AARON WILLIAM PERRY
ENVISION
Envision a global movement—including you, me and millions of others—mobilizing and scaling
soil-building and soil-stewardship activities in communities all over the planet.
Envision thousands of communities all over the world
➥ What daily practice will you choose to
with active, vibrant Soil Stewardship Guilds.
connect with soil?
➥ What soil stewardship activity are
...```

---

### Page 30

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (definitions, solutions, locations) but NO relationships were extracted

**Text Sample**:
```
SSSOOOIIILLL SSSTTTEEEWWWAAARRRDDDSSSHHHIIIPPP PPPLLLEEEDDDGGGEEE
I , , believe that
my connection with living soil is sacred. I promise to be a faithful
steward of soil, and thus of Mother Earth—through my direct
interactions with soil as well as the indirect infl uences of my
personal choices and consumer demand. I promise to be mindful
of my impact upon soil every day. I will compost all of my 
...```

---

### Page 32

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (benefits, locations) but NO relationships were extracted

**Text Sample**:
```
ACKNOWLEDGEMENTS
It truly takes a village to write and publish a book! This Soil Stewardship Handbook is no exception
and has benefited from the insights and expertise of so many dear friends and colleagues. A very
special thank you to:
Adrian Del Caro Marcia Perry
Aly Artusio-Glimpse Mark Bosco, SJ
Amelia Vincent Mark Guttridge
Artem Nikulkov Marty Sugg
Brad and Lindsay Lidge Meri Mullins
Brigitt
...```

---

### Page 33

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (statistics, causes, locations) but NO relationships were extracted

**Text Sample**:
```
20 AARON WILLIAM PERRY
NOTES
1 The Soil Stewardship Guild—a Y on Earth Community program, is sometimes referred to
by the letters “SSG” and is sometimes simply called the “Guild” for short.
2 As the table above indicates, Earth’s ocean contains an enormous amount of carbon—
dissolved in the water and taken up by phytoplankton and other life in the water. Although
the ocean contains far more carbon
...```

---

### Page 34

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (statistics, processes, problems, organizations, people, locations) but NO relationships were extracted
- Page has 7 knowledge indicators but only 0 relationships extracted - likely missing content

**Text Sample**:
```
REFERENCES
Ackerman-Leist, Philip. Rebuilding the Foodshed: How to Create Local, Sustainable, and Secure
Food Systems. White River Junction, VT: Chelsea Green Publishing, 2013.
Adenipekun, C.O. and R. Lawal. “Uses of Mushrooms in Bioremediation: A Review.”
Department of Botany, University of Ibadan. Ibadan, Nigeria, June 14, 2012. http://www.
academicjournals.org/article/article1380187476_Adenipek
...```

---

### Page 35

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (solutions, organizations, locations) but NO relationships were extracted

**Text Sample**:
```
22 AARON WILLIAM PERRY
Appelhof, Mary. Worms Eat My Garbage: How to Set Up and Maintain a Worm Composting
System, 2nd Edition. White River Junction, VT: Chelsea Green Pub., 2016 (Forthcoming).
Arava Institute for Environmental Studies. Kibbutz Ketura, Israel. http://arava.org/.
Backyard Gardening Blog. “How to Grow Amaranth.” http://www.gardeningblog.net/how-to-
grow/amaranth/. Accessed 1.24.17.
B
...```

---

### Page 37

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (organizations, locations) but NO relationships were extracted

**Text Sample**:
```
24 AARON WILLIAM PERRY
Creasy, Rosalind with Cathy Wilkinson Barash. “Edible Landscaping: Grow $700 of Food in 100
Square Feet!” Mother Earth News. Dec. 2009/Jan. 2010.
Dalai Lama XIV, The. “A Question of Our Own Survival.” In Moral Ground: Ethical Action for a
Planet in Peril. Forward by Desmond Tutu. San Antonio, TX: Trinity University Press, 2010.
———. Transforming the Mind: Teachings on Genera
...```

---

### Page 38

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (organizations, locations) but NO relationships were extracted

**Text Sample**:
```
SOIL STEWARDSHIP HANDBOOK • REFERENCES 25
Fisher, Adrian Ayres. “Why Not Start Today: Backyard Carbon Sequestration Is Something Nearly
Everyone Can Do.” Resilience: Building a World of Resilient Communities. Sept. 2, 2015.
Flores, H.C., with forward by Toby Hemenway and illustrations by Jackie Holmstrom. Food Not
Lawns: How to Turn Your Yard Into a Garden and Your Neighborhood Into a Community.
W
...```

---

### Page 40

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (statistics, solutions, organizations, locations) but NO relationships were extracted
- Page has 6 knowledge indicators but only 0 relationships extracted - likely missing content

**Text Sample**:
```
SOIL STEWARDSHIP HANDBOOK • REFERENCES 27
Independent. “The Human Brain is the Most Complex Structure in the Universe. Let’s Do All
We Can to Unravel Its Mysteries.” April 2, 2014. http://www.independent.co.uk/voices/
editorials/the-human-brain-is-the-most-complex-structure-in-the-universe-let-s-do-all-
we-can-to-unravel-its-9233125.html.
Ingham, Elaine and Carole Ann Rollins. 10 Steps to Gardenin
...```

---

### Page 42

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (definitions, locations) but NO relationships were extracted

**Text Sample**:
```
SOIL STEWARDSHIP HANDBOOK • REFERENCES 29
McIntosh, Alastair. Soil and Soul: People versus Corporate Power. London: Aurum Press, 2001.
McKibben, Bill. Earth: Making A Life On A Tough New Planet. New York: Time Books, 2010.
Miller, Kenneth. “How Mushrooms Can Save the World: Crusading Mycologist
Paul Stamets Says Fungi Can Clean Up Everything from Oil Spills to Nuclear
Meltdowns.” Discover Magazine
...```

---

### Page 43

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (definitions, problems, locations) but NO relationships were extracted

**Text Sample**:
```
30 AARON WILLIAM PERRY
Pimentel, David. “Soil Erosion: A Food and Environmental Threat.” Journal of the Environment,
Development and Sustainability. Vol. 8, 2006.
Pollan, Michael. Botany of Desire: A Plant’s Eye View of the World. New York: Random House,
2001.
Pope Francis. Laudato Si’: Encyclical on Climate Change and Inequality: On Care For Our
Common Home. Brooklyn, NY: Melville House Publishin
...```

---

### Page 45

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (processes, solutions, locations) but NO relationships were extracted

**Text Sample**:
```
32 AARON WILLIAM PERRY
Suzuki, David. The Legacy: An Elder’s Vision for Our Sustainable Future. Vancouver, BC:
Greystone Books, 2010.
———. Sacred Balance: Rediscovering Our Place in Nature. Vancouver, BC: Greystone, 1997.
Symphony of the Soil. Directed and Written by Deborah Koons, Perf. Ignacio Chapela, Vandana
Shiva, Elaine Ingham. 2013.
Tasch, Woody. Slow Money: Investing as if Food, Farms, and
...```

---

### Page 46

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (statistics, processes, solutions, people, locations) but NO relationships were extracted
- Page has 6 knowledge indicators but only 0 relationships extracted - likely missing content

**Text Sample**:
```
SOIL STEWARDSHIP HANDBOOK • REFERENCES 33
Urban Jungle. “Vertical Green Walls” http://www.urbanjunglephila.com/verticalgardens.html.
Accessed 12.12.16.
US Environmental Protection Agency. “Indoor Air Quality.” https://cfpub.epa.gov/roe/chapter/
air/indoorair.cfm. Accessed 12.12.16.
Vita, Ietef a.k.a. DJ CAVEM MOETEVATION. The Produce Section: The Harvest. Music Album.
http://djcavem.com/the-produc
...```

---

### Page 48

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (definitions, processes, benefits, organizations, locations) but NO relationships were extracted
- Page has 6 knowledge indicators but only 0 relationships extracted - likely missing content

**Text Sample**:
```
Y ON EARTH COMMUNITY
Th e Y on Earth Community is a network of diverse people and organizations
who curate and lead experiences, workshops and meet-ups that transform
our culture by cultivating thriving and sustainability practices. By delivering
as much inspiration as information and by having loads of fun while sharing
fresh, unique life-hacks, the Y on Earth Community helps people get smarter,

...```

---

### Page 49

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (definitions, locations) but NO relationships were extracted

**Text Sample**:
```
ABOUT Y ON EARTH
Y on Earth reveals how thriving by cultivating true health and well-being is
essential to creating a sustainable culture and future. Th is tour de force is
essential reading for millennials as well as their parents, grandparents, educators
and employers!
You can order copies of the eBook, audiobook and fi rst edition printed paperback
at www.yonearth.world/shop.
Here’s what people
...```

---

### Page 50

**Relationships Extracted**: 1

**Issues**:
- Page has 9 knowledge indicators but only 1 relationships extracted - likely missing content

**Text Sample**:
```
“Y on Earth is a needed read not only for Millennials but for all who want to survive—
both personally and globally. The current unsustainable consumption of time, goods
and the Earth’s collapsing resources threatens us all. Y on Earth is a pleasant, easy
read that assesses the systemic problems that face humanity, while offering insights
and approaches to reconnect to soil, to each other and to o
...```

---

### Page 51

**Relationships Extracted**: 0

**Issues**:
- Page has knowledge indicators (definitions, organizations, locations) but NO relationships were extracted

**Text Sample**:
```
ABOUT THE AUTHOR
Aaron William Perry is a writer, public speaker, impact entre-
preneur, consultant, artist and father. The author of Y on Earth:
Get Smarter, Feel Better, Heal the Planet, Aaron works with the
Y on Earth Community team and Impact Ambassadors to spread
the THRIVING & SUSTAINABILITY messages of hopeful and
empowering information and inspiration to diverse communities
throughout the 
...```

---

## Part 3: Pages Completely Skipped

**41** pages had NO extractions at all:

**Pages**: 1-2, 4-11, 13-14, 16-17, 20, 22-35, 37-38, 40, 42-43, 45-46, 48-49, 51-53

**Note**: These pages may contain:
- Front matter (title, copyright, table of contents)
- Images/diagrams without extractable text
- Chapter dividers
- OR actual content that was missed

## Recommendations

### Fix Incorrect Relationships

1. **Update extraction prompts** to address the 9 incorrect relationships
2. **Add validation rules** to catch these issues automatically
3. **Re-extract** after prompt improvements

### Improve Coverage

1. **Review 35 pages** flagged for missing knowledge
2. **Adjust chunking strategy** - some pages may have been partially chunked
3. **Lower extraction threshold** if filtering out too many valid relationships

### Increase Page Coverage

1. **Manually review** a sample of the 41 skipped pages
2. **Verify** if they contain extractable knowledge
3. **Adjust chunking or extraction logic** if valuable pages were missed
