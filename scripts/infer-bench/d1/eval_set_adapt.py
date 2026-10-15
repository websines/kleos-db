"""D1 adaptation set: 60 supersession items, disjoint from eval_set.py (new people, attributes and wording).

Used ONLY to pick prompts / thresholds / fit a probe. Never evaluated on as a test set.
(a = older fact, b = newer fact); label in {update, contradiction, compatible, duplicate}.
"""

ADAPT = [
    # --- updates (18)
    ("Nadia works as a barista at a cafe downtown.", "Nadia quit the cafe and is now a junior accountant at Deloitte.", "update"),
    ("Hugo lives in Toronto.", "Hugo relocated to Vancouver last autumn.", "update"),
    ("Owen is dating Clara.", "Owen and Clara broke up in the spring.", "update"),
    ("The user's dentist is Dr. Rossi.", "The user switched to a new dentist, Dr. Abebe, after moving.", "update"),
    ("Fatima rides a road bike to work.", "Fatima now takes the train to work since her bike was stolen.", "update"),
    ("The user's preferred editor is VS Code.", "The user has moved over to Neovim as their main editor.", "update"),
    ("Greta drinks coffee every morning.", "Greta gave up coffee and only drinks green tea now.", "update"),
    ("The launch is scheduled for March 1.", "The launch has been delayed to April 20.", "update"),
    ("Diego is married to Lucia.", "Diego and Lucia divorced last year.", "update"),
    ("The user's email is pat@oldmail.com.", "The user now uses pat@newmail.io; the oldmail address was closed.", "update"),
    ("Yuki is a junior designer.", "Yuki was promoted to lead designer in January.", "update"),
    ("The company hosts its website on Heroku.", "The company moved its website hosting from Heroku to Fly.io.", "update"),
    ("Ben's favourite sport is tennis.", "Ben says he is mostly into padel these days, not tennis.", "update"),
    ("Aisha owns a gray cat named Smokey.", "Smokey passed away and Aisha has no pets now.", "update"),
    ("The user works the night shift.", "The user moved to the day shift last week.", "update"),
    ("Leo's gym membership is at FitLife.", "Leo cancelled FitLife and joined a climbing gym instead.", "update"),
    ("The weekly meeting is on Tuesdays.", "The weekly meeting moved to Thursdays starting this month.", "update"),
    ("Ines is training for a half marathon.", "Ines stopped training after a knee injury and dropped the race.", "update"),
    # --- contradictions / corrections (6)
    ("Victor's anniversary is on October 12.", "Correction: Victor's anniversary is October 21, I mixed up the digits.", "contradiction"),
    ("The user's car is blue.", "The user's car is red.", "contradiction"),
    ("Sofia speaks three languages.", "Sofia actually speaks five languages.", "contradiction"),
    ("The server runs Ubuntu 20.04.", "The server is running Debian 12, not Ubuntu.", "contradiction"),
    ("Kenji is lactose intolerant.", "Kenji said he is not lactose intolerant; he just dislikes milk.", "contradiction"),
    ("The apartment rent is $1,800 per month.", "The apartment rent is $2,100 per month.", "contradiction"),
    # --- compatible additions (18)
    ("Nadia works as a barista at a cafe downtown.", "Nadia is taking an evening pottery class.", "compatible"),
    ("Hugo lives in Toronto.", "Hugo has a younger brother named Felix.", "compatible"),
    ("Owen is dating Clara.", "Owen and Clara are planning a trip to Japan.", "compatible"),
    ("The user's preferred editor is VS Code.", "The user writes most of their code in Go.", "compatible"),
    ("Greta drinks coffee every morning.", "Greta takes her coffee with oat milk.", "compatible"),
    ("Fatima rides a road bike to work.", "Fatima's office is about 12 kilometres from her home.", "compatible"),
    ("Diego is married to Lucia.", "Diego and Lucia have two daughters.", "compatible"),
    ("Yuki is a junior designer.", "Yuki mostly works on the mobile app's onboarding screens.", "compatible"),
    ("The company hosts its website on Heroku.", "The company's website gets about 50k visitors a month.", "compatible"),
    ("Ben's favourite sport is tennis.", "Ben plays chess online in the evenings.", "compatible"),
    ("Aisha owns a gray cat named Smokey.", "Aisha works from home three days a week.", "compatible"),
    ("The user works the night shift.", "The user is saving up for a new laptop.", "compatible"),
    ("Leo's gym membership is at FitLife.", "Leo goes to the gym before work, around 6 am.", "compatible"),
    ("The weekly meeting is on Tuesdays.", "The weekly meeting is usually run by Priscilla.", "compatible"),
    ("Victor's anniversary is on October 12.", "Victor is planning a surprise dinner for his anniversary.", "compatible"),
    ("Sofia speaks three languages.", "Sofia is learning to play the guitar.", "compatible"),
    ("The server runs Ubuntu 20.04.", "The server has 64 GB of RAM.", "compatible"),
    ("Kenji is lactose intolerant.", "Kenji loves spicy ramen.", "compatible"),
    # --- paraphrased duplicates (18)
    ("Nadia works as a barista at a cafe downtown.", "Nadia makes coffee at a downtown cafe for a living.", "duplicate"),
    ("Hugo lives in Toronto.", "Hugo's home is in Toronto, Canada.", "duplicate"),
    ("Owen is dating Clara.", "Clara is Owen's girlfriend.", "duplicate"),
    ("The user's dentist is Dr. Rossi.", "Dr. Rossi is the dentist the user sees.", "duplicate"),
    ("Fatima rides a road bike to work.", "Fatima commutes to her job on a road bicycle.", "duplicate"),
    ("The user's preferred editor is VS Code.", "The user likes Visual Studio Code best for editing.", "duplicate"),
    ("Greta drinks coffee every morning.", "Every morning Greta has a cup of coffee.", "duplicate"),
    ("The launch is scheduled for March 1.", "The launch date is the first of March.", "duplicate"),
    ("Diego is married to Lucia.", "Lucia is Diego's wife.", "duplicate"),
    ("Yuki is a junior designer.", "Yuki works as a designer at a junior level.", "duplicate"),
    ("The company hosts its website on Heroku.", "Our site is deployed on Heroku.", "duplicate"),
    ("Ben's favourite sport is tennis.", "Tennis is the sport Ben likes most.", "duplicate"),
    ("Aisha owns a gray cat named Smokey.", "Aisha has a grey cat called Smokey.", "duplicate"),
    ("The weekly meeting is on Tuesdays.", "We meet every Tuesday for the weekly sync.", "duplicate"),
    ("Victor's anniversary is on October 12.", "Victor celebrates his wedding anniversary on the 12th of October.", "duplicate"),
    ("The user's car is blue.", "The user drives a blue car.", "duplicate"),
    ("Sofia speaks three languages.", "Sofia is trilingual.", "duplicate"),
    ("The apartment rent is $1,800 per month.", "Monthly rent for the apartment is 1800 dollars.", "duplicate"),
]

assert len(ADAPT) == 60
