# == Schema Information
#
# Table name: images
#
#  id         :integer         not null, primary key
#  filepath   :string(255)
#  genotype	  :string(255)
#  created_at :datetime
#  updated_at :datetime
#

class Image < ActiveRecord::Base
end
