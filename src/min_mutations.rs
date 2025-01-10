use std::collections::{HashSet, VecDeque};

fn min_mutations(start_gene: &str, end_gene: &str, bank: Vec<&str>) -> i32 {
   let bank_set: HashSet<&str> = bank.into_iter().collect();

   if !bank_set.contains(end_gene) {
       return -1;
   }

   let mut queue: VecDeque<(String, i32)> = VecDeque::new();
   let mut visited: HashSet<String> = HashSet::new();

   queue.push_back((start_gene.to_string(), 0));
   visited.insert(start_gene.to_string());

   let bases: Vec<char> = vec!['A','C','G','T'];

   while let Some((current_gene, mutations)) = queue.pop_front() {
      if current_gene == end_gene {
          return mutations;
      }

      let current_gene_vec: Vec<char> = current_gene.chars().collect();

      for i in 0..current_gene_vec.len() {
          for &base in &bases {
	      if base != current_gene_vec[i] {
	         let mut new_gene_vec = current_gene_vec.clone();

		 new_gene_vec[i] = base;

		 let new_gene: String = new_gene_vec.into_iter().collect();

		 if bank_set.contains(new_gene.as_str()) && !visited.contains(new_gene.as_str()) {
		    queue.push_back((new_gene.clone(), mutations + 1));
		    visited.insert(new_gene);
		 }
	      }
	  }
      }
   }

   return -1;
}

#[cfg(test)]
mod tests {
    // RUSTFLAGS="-Awarnings --cfg debug_statements" cargo test test_min_mutations -- --nocapture
    use super::*;

    #[test]
    fn test_min_mutations_one() {
       let start_gene = "AACCGGTT";
       let end_gene = "AACCGGTA";

       let bank = vec!["AACCGGTA"];
       let result = min_mutations(start_gene, end_gene, bank);

       assert_eq!(result, 1);
    }
    
    #[test]
    fn test_min_mutations_two() {
       let start_gene = "AACCGGTT";
       let end_gene = "AAACGGTA";

       let bank = vec!["AACCGGTA", "AACCGCTA", "AAACGGTA"];
       let result = min_mutations(start_gene, end_gene, bank);

       assert_eq!(result, 2);
    }
}
