from typing import List

class Solution:
    def findDifference(self, nums1: List[int], nums2: List[int]) -> List[List[int]]:
        set1 = set(nums1)
        set2 = set(nums2)
        ans1 = []
        ans2 = []
        for v in nums1:
            if v not in set2:
                ans1.append(v)
        for v in nums2:
            if v not in set1:
                ans2.append(v)
        return [ans1, ans2]

if __name__ == "__main__":
    solution = Solution()
    
    # 测试用例1
    nums1 = [1, 2, 3]
    nums2 = [2, 4, 6]
    result1 = solution.findDifference(nums1, nums2)
    print(f"测试用例1: nums1={nums1}, nums2={nums2}")
    print(f"结果: {result1}")
    print(f"nums1中不在nums2的元素: {result1[0]}")
    print(f"nums2中不在nums1的元素: {result1[1]}")
    print()
    
    # 测试用例2
    nums3 = [1, 2, 3, 3]
    nums4 = [1, 1, 2, 2]
    result2 = solution.findDifference(nums3, nums4)
    print(f"测试用例2: nums1={nums3}, nums2={nums4}")
    print(f"结果: {result2}")
    print(f"nums1中不在nums2的元素: {result2[0]}")
    print(f"nums2中不在nums1的元素: {result2[1]}")
    print()
    
    # 测试用例3 - 空数组
    nums5 = []
    nums6 = [1, 2, 3]
    result3 = solution.findDifference(nums5, nums6)
    print(f"测试用例3: nums1={nums5}, nums2={nums6}")
    print(f"结果: {result3}")

