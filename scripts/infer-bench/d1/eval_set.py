"""D1 eval set: memory-operation decisions.

kind=supersede: (a = older fact, b = newer fact)
  label in {update, contradiction, compatible, duplicate}
  binary target "does B supersede A?" = label in {update, contradiction}
kind=entity: (a, b = two mentions with context); label in {same, different}
"""

SUPERSEDE = [
    # --- updates: job / location / relationship / status / preference changes
    ("Maya works as a nurse at St. Mary's Hospital.", "Maya started a new job as a data analyst at Stripe last month.", "update"),
    ("Tom lives in Berlin.", "Tom moved to Lisbon in March and has settled in there.", "update"),
    ("Priya is engaged to Arjun.", "Priya and Arjun called off their engagement.", "update"),
    ("The user's manager is Karen Liu.", "The user now reports to David Okafor after the reorg.", "update"),
    ("Sam drives a 2015 Honda Civic.", "Sam sold the Civic and bought a Tesla Model 3.", "update"),
    ("The user prefers Python for scripting.", "The user says they have switched to Rust for all their scripting now.", "update"),
    ("Lena is a vegetarian.", "Lena started eating meat again this year.", "update"),
    ("The project deadline is June 30.", "The project deadline was pushed to August 15.", "update"),
    ("Carlos is single.", "Carlos got married to Ana in September.", "update"),
    ("The user's phone number is 555-0134.", "The user's new phone number is 555-0199; the old one is disconnected.", "update"),
    ("Jin is a student at MIT.", "Jin graduated from MIT and now works at a startup in Boston.", "update"),
    ("The team uses Jira for issue tracking.", "The team migrated from Jira to Linear in Q2.", "update"),
    ("Ravi's favourite band is Coldplay.", "Ravi says his favourite band these days is Radiohead.", "update"),
    ("Emma rents an apartment in Brooklyn.", "Emma bought a house in New Jersey and moved out of Brooklyn.", "update"),
    # --- contradictions / corrections of the same attribute
    ("Alice's birthday is May 3.", "Correction: Alice's birthday is actually June 3, not May 3.", "contradiction"),
    ("The office is on the 4th floor.", "The office is on the 7th floor.", "contradiction"),
    ("The user has two children.", "The user mentioned they have three kids.", "contradiction"),
    ("Mark is allergic to peanuts.", "Mark clarified he is not allergic to peanuts; it was shellfish all along.", "contradiction"),
    # --- compatible additions (both remain true)
    ("Maya works as a nurse at St. Mary's Hospital.", "Maya volunteers at an animal shelter on weekends.", "compatible"),
    ("Tom lives in Berlin.", "Tom is learning to play the cello.", "compatible"),
    ("Priya is engaged to Arjun.", "Priya and Arjun adopted a dog named Biscuit.", "compatible"),
    ("The user prefers Python for scripting.", "The user also uses TypeScript for frontend work.", "compatible"),
    ("Lena is a vegetarian.", "Lena loves Italian food, especially risotto.", "compatible"),
    ("Sam drives a 2015 Honda Civic.", "Sam commutes 40 minutes to work each day.", "compatible"),
    ("Carlos is single.", "Carlos enjoys rock climbing.", "compatible"),
    ("Jin is a student at MIT.", "Jin is majoring in mechanical engineering.", "compatible"),
    ("The team uses Jira for issue tracking.", "The team holds stand-ups every morning at 9:30.", "compatible"),
    ("Emma rents an apartment in Brooklyn.", "Emma's apartment has a small balcony with plants.", "compatible"),
    ("The user's manager is Karen Liu.", "Karen Liu also manages the design team.", "compatible"),
    ("Ravi's favourite band is Coldplay.", "Ravi went to a Coldplay concert in London last year.", "compatible"),
    # --- paraphrased duplicates (same fact restated)
    ("Maya works as a nurse at St. Mary's Hospital.", "Maya is employed at St. Mary's Hospital as a nurse.", "duplicate"),
    ("Tom lives in Berlin.", "Tom's home is in Berlin, Germany.", "duplicate"),
    ("Priya is engaged to Arjun.", "Arjun and Priya are engaged to be married.", "duplicate"),
    ("The user prefers Python for scripting.", "For scripts, the user likes to use Python.", "duplicate"),
    ("Lena is a vegetarian.", "Lena doesn't eat meat.", "duplicate"),
    ("The project deadline is June 30.", "The project is due at the end of June.", "duplicate"),
    ("Sam drives a 2015 Honda Civic.", "Sam's car is a Honda Civic from 2015.", "duplicate"),
    ("Jin is a student at MIT.", "Jin studies at the Massachusetts Institute of Technology.", "duplicate"),
    ("The team uses Jira for issue tracking.", "Issues on the team are tracked in Jira.", "duplicate"),
    ("Alice's birthday is May 3.", "Alice was born on the 3rd of May.", "duplicate"),
    ("The office is on the 4th floor.", "Our office is located on floor four.", "duplicate"),
    ("Mark is allergic to peanuts.", "Mark has a peanut allergy.", "duplicate"),
]

ENTITY = [
    # same entity
    ("Bob Smith, a backend engineer at Acme, mentioned the outage.", "Robert Smith from Acme's backend team fixed the database.", "same"),
    ("Dr. Elena Garcia is the user's cardiologist.", "The user's heart doctor, Elena Garcia, ordered an ECG.", "same"),
    ("NYC is where the conference takes place.", "The conference venue is in New York City.", "same"),
    ("The user's sister Kate lives in Denver.", "Katherine, the user's younger sister, visited from Denver.", "same"),
    ("Microsoft announced new Copilot features.", "MSFT shares rose after the Copilot launch.", "same"),
    ("Liz from accounting approved the invoice.", "Elizabeth Moore in accounting signed off on the invoice.", "same"),
    ("The user's dog, Max, is a golden retriever.", "Max the retriever needs his vet appointment on Friday.", "same"),
    ("JFK airport had delays this morning.", "John F. Kennedy International Airport reported long delays.", "same"),
    ("Prof. Chen teaches the user's ML course.", "Wei Chen, the machine learning professor, posted the grades.", "same"),
    ("The user's employer, Globex Corp, is hiring.", "Globex Corporation, where the user works, opened new roles.", "same"),
    ("Mom called about Sunday dinner.", "The user's mother, Linda, is hosting dinner on Sunday.", "same"),
    ("The repo kleos-db had a new release.", "kleos (the database repository) shipped version 0.18.", "same"),
    # different entities
    ("Paris is the capital of France.", "Paris Hilton launched a new fragrance.", "different"),
    ("Apple released a new iPhone.", "The user ate an apple for breakfast.", "different"),
    ("John Smith, the user's dentist, is in Seattle.", "John Smith, the user's college roommate, lives in Austin.", "different"),
    ("Jordan, the user's coworker, is on vacation.", "The user wants to travel to Jordan to see Petra.", "different"),
    ("Amazon delivered the package late.", "The Amazon rainforest is losing tree cover.", "different"),
    ("The user's son Alex plays soccer.", "Alex, the user's landlord, raised the rent.", "different"),
    ("Mercury is the closest planet to the Sun.", "Freddie Mercury was the lead singer of Queen.", "different"),
    ("Kate, the user's sister, lives in Denver.", "Kate, the user's manager, lives in Chicago.", "different"),
    ("Python is the user's main programming language.", "A python escaped from the city zoo.", "different"),
    ("Washington state has strict hiking permits.", "George Washington was the first US president.", "different"),
    ("Dr. Patel is the user's dermatologist.", "Dr. Patel, the user's PhD advisor, reviewed the thesis.", "different"),
    ("The user's cat Luna is sick.", "Luna, the user's niece, turned five today.", "different"),
]
