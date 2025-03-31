import time
from typing import List


class Solution:
    def populate_buffer(self, buffer_size, processed_idx, buff, nums):
        """
        Populate a buffer of given size and return the idxs processed
        to do so.
        """
        for ii, num in enumerate(nums):
            if ii <= processed_idx:
                continue

            if num in buff:
                buff.remove(num)
            else:
                buff.append(num)

                if len(buff) == buffer_size:
                    return ii, buff

        # NB cannot fill the buffer
        return -1, buff

    def filter_buffer(self, processed_idx, buff, nums):
        for ii, num in enumerate(nums):
            if ii <= processed_idx:
                continue

            if num in buff:
                buff.remove(num)
                
        return buff

    def singleNumber(self, nums: List[int]) -> int:
        buffer_size, processed_idx, buff = 10, 0, [nums[0]]

        while processed_idx < len(nums):
            processed_idx, buff = self.populate_buffer(
                buffer_size, processed_idx, buff, nums
            )

            print(f"Populated buffer to {buff} for idx={processed_idx}")

            buff, nums = self.filter_buffer(processed_idx, buff, nums)

            print(f"Filtered buffer to {buff}")

            time.sleep(2)
            
            if len(buff) == 0:
                continue
                        
            if len(buff) == 1:
                # NB found the answer.
                return buff[0]

            """
            if processed_idx == -1:
                # NB not sufficient numbers to fill the buffer, reset.
                buffer_size /= 2
                processed_idx = 0
                buff = [nums[0]]

                print(f"Shortening buffer to size {buffer_size} ")
            """
            
if __name__ == "__main__":
    nums = [
        -967,
        -590,
        980,
        -806,
        145,
        -881,
        357,
        -787,
        -592,
        859,
        627,
        -929,
        296,
        818,
        -194,
        586,
        -106,
        -479,
        967,
        132,
        -396,
        -692,
        464,
        726,
        -967,
        -590,
        980,
        -806,
        145,
        -881,
        357,
        -787,
        -592,
        859,
        627,
        -929,
        296,
        818,
        -194,
        586,
        -106,
        -479,
        967,
        132,
        -396,
        -692,
        464,
        726,
        -755,
    ]

    print(f"Solving for {len(nums)} elements of {nums}")

    result = Solution().singleNumber(nums)

    print(f"Found result: {result}")
